#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

static int write_full(int fd, const void *data, size_t size)
{
    const unsigned char *cursor = data;
    size_t done = 0;

    while (done < size) {
        ssize_t result = send(fd, cursor + done, size - done, MSG_NOSIGNAL);

        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return result < 0 ? -errno : -EPIPE;
        done += (size_t)result;
    }
    return 0;
}

static int read_full(int fd, void *data, size_t size)
{
    unsigned char *cursor = data;
    size_t done = 0;

    while (done < size) {
        ssize_t result = recv(fd, cursor + done, size - done, 0);

        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return result < 0 ? -errno : -ECONNRESET;
        done += (size_t)result;
    }
    return 0;
}

void npu_ipc_header_init(npu_ipc_header_t *header, uint16_t opcode,
                         uint64_t request_id, uint64_t object_id,
                         uint32_t payload_size)
{
    memset(header, 0, sizeof(*header));
    header->magic = NPU_IPC_MAGIC;
    header->version = NPU_IPC_VERSION;
    header->header_size = sizeof(*header);
    header->opcode = opcode;
    header->payload_size = payload_size;
    header->request_id = request_id;
    header->object_id = object_id;
}

static int make_unix_address(const char *path, struct sockaddr_un *address,
                             socklen_t *size_out)
{
    size_t length;

    if (!path || !*path || !address || !size_out)
        return -EINVAL;
    length = strlen(path);
    if (length >= sizeof(address->sun_path))
        return -ENAMETOOLONG;
    memset(address, 0, sizeof(*address));
    address->sun_family = AF_UNIX;
    memcpy(address->sun_path, path, length + 1);
    *size_out = (socklen_t)(offsetof(struct sockaddr_un, sun_path) +
                            length + 1);
    return 0;
}

int npu_ipc_connect(const char *path, uint32_t timeout_ms)
{
    struct sockaddr_un address;
    socklen_t address_size;
    int fd;
    int flags;
    int status;

    status = make_unix_address(path, &address, &address_size);
    if (status < 0)
        return status;
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -errno;
    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        status = -errno;
        goto fail;
    }
    if (connect(fd, (const struct sockaddr *)&address, address_size) < 0) {
        struct pollfd pollfd = { .fd = fd, .events = POLLOUT };
        int result;

        if (errno != EINPROGRESS) {
            status = -errno;
            goto fail;
        }
        do {
            result = poll(&pollfd, 1,
                          timeout_ms == UINT32_MAX ? -1 : (int)timeout_ms);
        } while (result < 0 && errno == EINTR);
        if (result == 0) {
            status = -ETIMEDOUT;
            goto fail;
        }
        if (result < 0) {
            status = -errno;
            goto fail;
        }
        {
            int error = 0;
            socklen_t error_size = sizeof(error);

            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &error_size) < 0) {
                status = -errno;
                goto fail;
            }
            if (error) {
                status = -error;
                goto fail;
            }
        }
    }
    if (fcntl(fd, F_SETFL, flags) < 0) {
        status = -errno;
        goto fail;
    }
    return fd;

fail:
    close(fd);
    return status;
}

int npu_ipc_listen(const char *path, unsigned backlog)
{
    struct sockaddr_un address;
    socklen_t address_size;
    int fd;
    int status;

    status = make_unix_address(path, &address, &address_size);
    if (status < 0)
        return status;
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -errno;
    if (unlink(path) < 0 && errno != ENOENT) {
        status = -errno;
        goto fail;
    }
    if (bind(fd, (const struct sockaddr *)&address, address_size) < 0) {
        status = -errno;
        goto fail;
    }
    if (chmod(path, 0660) < 0) {
        status = -errno;
        goto fail_unlink;
    }
    if (listen(fd, backlog ? (int)backlog : 32) < 0) {
        status = -errno;
        goto fail_unlink;
    }
    return fd;

fail_unlink:
    (void)unlink(path);
fail:
    close(fd);
    return status;
}

int npu_ipc_send(int fd, const npu_ipc_header_t *header,
                 const void *payload, const int *handles,
                 size_t handle_count)
{
    unsigned char control[CMSG_SPACE(sizeof(int) * NPU_HAL_DATA_MAX_HANDLES)];
    struct msghdr message;
    struct iovec vector;
    ssize_t sent;
    int status;

    if (fd < 0 || !header || (!payload && header->payload_size) ||
        handle_count > NPU_HAL_DATA_MAX_HANDLES ||
        (handle_count && !handles) ||
        header->magic != NPU_IPC_MAGIC ||
        header->version != NPU_IPC_VERSION ||
        header->header_size != sizeof(*header) ||
        header->payload_size > NPU_IPC_MAX_PAYLOAD)
        return -EINVAL;
    memset(&message, 0, sizeof(message));
    vector.iov_base = (void *)header;
    vector.iov_len = sizeof(*header);
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    if (handle_count) {
        struct cmsghdr *control_header;

        memset(control, 0, sizeof(control));
        message.msg_control = control;
        message.msg_controllen = CMSG_SPACE(sizeof(int) * handle_count);
        control_header = CMSG_FIRSTHDR(&message);
        control_header->cmsg_level = SOL_SOCKET;
        control_header->cmsg_type = SCM_RIGHTS;
        control_header->cmsg_len = CMSG_LEN(sizeof(int) * handle_count);
        memcpy(CMSG_DATA(control_header), handles,
               sizeof(int) * handle_count);
    }
    do {
        sent = sendmsg(fd, &message, MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    if (sent < 0)
        return -errno;
    if (!sent)
        return -EPIPE;
    if ((size_t)sent < sizeof(*header)) {
        status = write_full(fd, (const unsigned char *)header + sent,
                            sizeof(*header) - (size_t)sent);
        if (status < 0)
            return status;
    }
    return header->payload_size ?
        write_full(fd, payload, header->payload_size) : 0;
}

static void close_handles(int *handles, size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        if (handles[index] >= 0)
            close(handles[index]);
        handles[index] = -1;
    }
}

int npu_ipc_receive(int fd, npu_ipc_header_t *header, void **payload_out,
                    int handles[NPU_HAL_DATA_MAX_HANDLES],
                    size_t *handle_count_out)
{
    unsigned char control[CMSG_SPACE(sizeof(int) * NPU_HAL_DATA_MAX_HANDLES)];
    struct msghdr message;
    struct iovec vector;
    size_t handle_count = 0;
    size_t received_size;
    void *payload = 0;
    ssize_t received;
    int status;
    size_t index;

    if (fd < 0 || !header || !payload_out || !handles || !handle_count_out)
        return -EINVAL;
    *payload_out = 0;
    *handle_count_out = 0;
    for (index = 0; index < NPU_HAL_DATA_MAX_HANDLES; ++index)
        handles[index] = -1;
    memset(&message, 0, sizeof(message));
    memset(control, 0, sizeof(control));
    vector.iov_base = header;
    vector.iov_len = sizeof(*header);
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    do {
        received = recvmsg(fd, &message, MSG_CMSG_CLOEXEC);
    } while (received < 0 && errno == EINTR);
    if (received < 0)
        return -errno;
    if (!received)
        return -ECONNRESET;
    received_size = (size_t)received;
    {
        struct cmsghdr *control_header;

        for (control_header = CMSG_FIRSTHDR(&message); control_header;
             control_header = CMSG_NXTHDR(&message, control_header)) {
            size_t bytes;
            size_t count;

            if (control_header->cmsg_level != SOL_SOCKET ||
                control_header->cmsg_type != SCM_RIGHTS ||
                control_header->cmsg_len < CMSG_LEN(0))
                continue;
            bytes = control_header->cmsg_len - CMSG_LEN(0);
            count = bytes / sizeof(int);
            if (count > NPU_HAL_DATA_MAX_HANDLES - handle_count) {
                close_handles(handles, handle_count);
                return -EMSGSIZE;
            }
            memcpy(handles + handle_count, CMSG_DATA(control_header),
                   count * sizeof(int));
            handle_count += count;
        }
    }
    if (message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) {
        close_handles(handles, handle_count);
        return -EMSGSIZE;
    }
    if (received_size < sizeof(*header)) {
        status = read_full(fd, (unsigned char *)header + received_size,
                           sizeof(*header) - received_size);
        if (status < 0) {
            close_handles(handles, handle_count);
            return status;
        }
    }
    if (header->magic != NPU_IPC_MAGIC ||
        header->version != NPU_IPC_VERSION ||
        header->header_size != sizeof(*header) || header->reserved ||
        header->payload_size > NPU_IPC_MAX_PAYLOAD) {
        close_handles(handles, handle_count);
        return -EPROTO;
    }
    if (header->payload_size) {
        payload = malloc(header->payload_size);
        if (!payload) {
            close_handles(handles, handle_count);
            return -ENOMEM;
        }
        status = read_full(fd, payload, header->payload_size);
        if (status < 0) {
            free(payload);
            close_handles(handles, handle_count);
            return status;
        }
    }
    *payload_out = payload;
    *handle_count_out = handle_count;
    return 0;
}
