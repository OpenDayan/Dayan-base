#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Mangle.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

#ifndef DYCC_SOURCE_ROOT
#define DYCC_SOURCE_ROOT "."
#endif

namespace fs = std::filesystem;
using clang::AnnotateAttr;
using clang::ASTContext;
using clang::CharSourceRange;
using clang::CompilerInstance;
using clang::Decl;
using clang::DiagnosticsEngine;
using clang::FunctionDecl;
using clang::GlobalDecl;
using clang::LangOptions;
using clang::Lexer;
using clang::MangleContext;
using clang::ParmVarDecl;
using clang::PrintingPolicy;
using clang::QualType;
using clang::RecursiveASTVisitor;
using clang::SourceLocation;
using clang::SourceManager;
using clang::SourceRange;
using clang::Token;
using clang::VarDecl;

namespace {

enum class Side {
    Both,
    Host,
    Device,
    Kernel,
    Invalid,
};

struct Edit {
    std::size_t begin;
    std::size_t end;
    std::string replacement;
};

struct SplitOutput {
    std::string host;
    std::string device;
    bool failed = false;
};

struct Options {
    fs::path input;
    fs::path output;
    fs::path root;
    fs::path host_cxx = "clang++";
    fs::path device_cxx = "clang++";
    fs::path objcopy = "llvm-objcopy";
    fs::path save_temps_dir;
    fs::path emit_host;
    fs::path emit_device;
    std::vector<std::string> common_args;
    std::vector<std::string> host_args;
    std::vector<std::string> device_args;
    std::vector<fs::path> source_inputs;
    std::vector<std::string> link_args;
    bool save_temps = false;
    bool split_only = false;
    bool compile_only = false;
    bool dry_run = false;
    bool verbose = false;
};

static void print_usage(std::ostream &out)
{
    out <<
        "usage: dycc [options] -c input.cpp [-o output.o]\n"
        "       dycc [options] input.cpp... [objects/libraries] [-o program]\n"
        "\n"
        "Dayan compiler-driver options:\n"
        "  -c                       compile to Host object without linking\n"
        "  --dayan-root <dir>       repository/SDK root\n"
        "  --host-cxx <path>        Host clang++ (default: clang++)\n"
        "  --device-cxx <path>      Device clang++ (default: clang++)\n"
        "  -Xhost <arg>             pass one argument only to Host clang\n"
        "  -Xdevice <arg>           pass one argument only to Device clang\n"
        "  -Xlinker <arg>           pass one argument to the Host linker\n"
        "  --save-temps[=<dir>]     retain generated sources and objects\n"
        "  --emit-host <path>       also write the split Host source\n"
        "  --emit-device <path>     also write the split Device source\n"
        "  --split-only             stop after source generation\n"
        "  --dry-run                print compiler commands without running\n"
        "  -v, --verbose            print executed compiler commands\n"
        "  --version                print dycc version\n"
        "\n"
        "Without -c, dycc links a Host executable with libDayanRuntime.\n"
        "Unrecognised compiler options are forwarded to both clang passes.\n";
}

static bool starts_with(const std::string &value, const std::string &prefix)
{
    return value.compare(0, prefix.size(), prefix) == 0;
}

static bool is_dayan_root(const fs::path &root)
{
    return fs::is_directory(root / "include/api") &&
           fs::is_directory(root / "host/include") &&
           fs::is_regular_file(
               root / "host/dycc/include/dayan_compiler.h");
}

static fs::path default_dayan_root(const char *argv0)
{
    std::error_code error;
    fs::path executable = fs::read_symlink("/proc/self/exe", error);
    if (error && argv0 && fs::path(argv0).has_parent_path()) {
        error.clear();
        executable = fs::weakly_canonical(argv0, error);
    }
    if (!error && !executable.empty()) {
        fs::path prefix = executable.parent_path().parent_path();
        const fs::path candidates[] = {
            prefix / "lib/dayan",
            prefix / "lib64/dayan",
            prefix / "share/dayan",
        };
        for (const fs::path &candidate : candidates) {
            if (is_dayan_root(candidate))
                return candidate;
        }
    }
    return fs::path(DYCC_SOURCE_ROOT);
}

static std::optional<std::string> take_value(int &index, int argc,
                                             char **argv,
                                             const std::string &option)
{
    if (index + 1 >= argc) {
        std::cerr << "dycc: error: missing value after " << option << "\n";
        return std::nullopt;
    }
    return std::string(argv[++index]);
}

static bool compiler_option_takes_value(const std::string &arg)
{
    static const std::set<std::string> options = {
        "-I", "-D", "-U", "-include", "-imacros", "-isystem",
        "-iquote", "-idirafter", "-isysroot", "--sysroot", "-x",
        "-std", "-stdlib", "-resource-dir", "-gcc-toolchain",
        "--gcc-toolchain", "-fdebug-prefix-map", "-ffile-prefix-map",
        "-fmacro-prefix-map", "-MF", "-MT", "-MQ",
    };
    return options.count(arg) != 0;
}

static bool is_link_input(const fs::path &path)
{
    std::string name = path.filename().string();
    std::string extension = path.extension().string();
    return extension == ".o" || extension == ".obj" || extension == ".a" ||
           extension == ".so" || extension == ".dylib" ||
           name.find(".so.") != std::string::npos;
}

static bool is_link_only_option(const std::string &arg)
{
    return arg == "-L" || arg == "-l" || arg == "-T" || arg == "-u" ||
           arg == "-z" || arg == "-Xlinker" || arg == "-static" ||
           arg == "-static-libgcc" || arg == "-static-libstdc++" ||
           arg == "-pie" || arg == "-no-pie" || arg == "-rdynamic" ||
           arg == "-s" || starts_with(arg, "-L") ||
           starts_with(arg, "-l") || starts_with(arg, "-Wl,") ||
           starts_with(arg, "-fuse-ld=");
}

static bool link_option_takes_value(const std::string &arg)
{
    return arg == "-L" || arg == "-l" || arg == "-T" || arg == "-u" ||
           arg == "-z" || arg == "-Xlinker";
}

static bool parse_options(int argc, char **argv, Options &options)
{
    std::vector<fs::path> inputs;
    bool positional_only = false;

    options.root = default_dayan_root(argc > 0 ? argv[0] : nullptr);
    if (const char *root = std::getenv("DAYAN_ROOT"))
        options.root = root;
    if (const char *host = std::getenv("DYCC_HOST_CXX"))
        options.host_cxx = host;
    if (const char *device = std::getenv("DYCC_DEVICE_CXX"))
        options.device_cxx = device;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (!positional_only && (arg == "-h" || arg == "--help")) {
            print_usage(std::cout);
            std::exit(0);
        }
        if (!positional_only && arg == "--version") {
            std::cout << "dycc 0.1 (Dayan clang driver)\n";
            std::exit(0);
        }
        if (!positional_only && arg == "--") {
            positional_only = true;
            continue;
        }
        if (!positional_only && arg == "-c") {
            options.compile_only = true;
            continue;
        }
        if (!positional_only && (arg == "-v" || arg == "--verbose")) {
            options.verbose = true;
            continue;
        }
        if (!positional_only && arg == "--dry-run") {
            options.dry_run = true;
            continue;
        }
        if (!positional_only && arg == "--split-only") {
            options.split_only = true;
            continue;
        }
        if (!positional_only && arg == "--save-temps") {
            options.save_temps = true;
            continue;
        }
        if (!positional_only && starts_with(arg, "--save-temps=")) {
            options.save_temps = true;
            options.save_temps_dir = arg.substr(std::strlen("--save-temps="));
            continue;
        }
        auto path_option = [&](const char *name, fs::path &destination) {
            if (arg != name)
                return false;
            auto value = take_value(i, argc, argv, arg);
            if (!value) {
                destination.clear();
                return true;
            }
            destination = *value;
            return true;
        };
        if (!positional_only && path_option("--dayan-root", options.root)) {
            if (options.root.empty())
                return false;
            continue;
        }
        if (!positional_only && path_option("--host-cxx", options.host_cxx)) {
            if (options.host_cxx.empty())
                return false;
            continue;
        }
        if (!positional_only && path_option("--device-cxx", options.device_cxx)) {
            if (options.device_cxx.empty())
                return false;
            continue;
        }
        if (!positional_only && path_option("--objcopy", options.objcopy)) {
            if (options.objcopy.empty())
                return false;
            continue;
        }
        if (!positional_only && path_option("--emit-host", options.emit_host)) {
            if (options.emit_host.empty())
                return false;
            continue;
        }
        if (!positional_only && path_option("--emit-device", options.emit_device)) {
            if (options.emit_device.empty())
                return false;
            continue;
        }
        if (!positional_only && (arg == "-o" || arg == "--output")) {
            auto value = take_value(i, argc, argv, arg);
            if (!value)
                return false;
            options.output = *value;
            continue;
        }
        if (!positional_only && starts_with(arg, "-o") && arg.size() > 2) {
            options.output = arg.substr(2);
            continue;
        }
        if (!positional_only && (arg == "-Xhost" || arg == "-Xdevice")) {
            auto value = take_value(i, argc, argv, arg);
            if (!value)
                return false;
            (arg == "-Xhost" ? options.host_args : options.device_args)
                .push_back(*value);
            continue;
        }
        if (!positional_only && is_link_only_option(arg)) {
            options.link_args.push_back(arg);
            if (link_option_takes_value(arg)) {
                auto value = take_value(i, argc, argv, arg);
                if (!value)
                    return false;
                options.link_args.push_back(*value);
            }
            continue;
        }
        if (!positional_only && starts_with(arg, "--host-arg=")) {
            options.host_args.push_back(arg.substr(std::strlen("--host-arg=")));
            continue;
        }
        if (!positional_only && starts_with(arg, "--device-arg=")) {
            options.device_args.push_back(
                arg.substr(std::strlen("--device-arg=")));
            continue;
        }
        if (!positional_only && !arg.empty() && arg[0] == '-') {
            if (arg == "-S" || arg == "-E" || arg == "-shared" ||
                arg == "--target" || starts_with(arg, "--target=") ||
                arg == "-target" || starts_with(arg, "-march=") ||
                starts_with(arg, "-mabi=")) {
                std::cerr << "dycc: error: unsupported or fixed driver option '"
                          << arg << "'\n";
                return false;
            }
            options.common_args.push_back(arg);
            if (compiler_option_takes_value(arg)) {
                auto value = take_value(i, argc, argv, arg);
                if (!value)
                    return false;
                options.common_args.push_back(*value);
            }
            continue;
        }
        if (is_link_input(arg))
            options.link_args.push_back(arg);
        else
            inputs.emplace_back(arg);
    }

    if (inputs.empty()) {
        std::cerr << "dycc: error: at least one input source is required\n";
        return false;
    }
    options.source_inputs = std::move(inputs);
    if (options.compile_only && !options.link_args.empty()) {
        std::cerr << "dycc: error: linker inputs/options cannot be used with -c\n";
        return false;
    }
    if ((options.split_only || !options.emit_host.empty() ||
         !options.emit_device.empty()) && options.source_inputs.size() != 1) {
        std::cerr << "dycc: error: split/emit options require exactly one "
                     "input source\n";
        return false;
    }
    if (options.compile_only && options.source_inputs.size() != 1 &&
        !options.output.empty()) {
        std::cerr << "dycc: error: cannot use -o with -c and multiple inputs\n";
        return false;
    }
    if (options.output.empty() && !options.split_only) {
        if (options.compile_only && options.source_inputs.size() == 1) {
            options.output = options.source_inputs.front().filename();
            options.output.replace_extension(".o");
        } else if (!options.compile_only) {
            options.output = "a.out";
        }
    }
    if (options.split_only && !options.save_temps && options.emit_host.empty() &&
        options.emit_device.empty()) {
        std::cerr << "dycc: error: --split-only needs --save-temps or an "
                     "--emit-* output\n";
        return false;
    }
    return true;
}

static Side annotation_side(const Decl *decl)
{
    Side result = Side::Both;
    for (const auto *attr : decl->specific_attrs<AnnotateAttr>()) {
        Side current = Side::Both;
        if (attr->getAnnotation() == "dayan.host")
            current = Side::Host;
        else if (attr->getAnnotation() == "dayan.device")
            current = Side::Device;
        else if (attr->getAnnotation() == "dayan.kernel")
            current = Side::Kernel;
        if (current == Side::Both)
            continue;
        if (result != Side::Both && result != current)
            return Side::Invalid;
        result = current;
    }
    return result;
}

static Side function_side(const FunctionDecl *decl)
{
    Side result = Side::Both;
    for (const FunctionDecl *redecl : decl->redecls()) {
        Side current = annotation_side(redecl);
        if (current == Side::Invalid)
            return current;
        if (current == Side::Both)
            continue;
        if (result != Side::Both && result != current)
            return Side::Invalid;
        result = current;
    }
    return result;
}

class SplitVisitor : public RecursiveASTVisitor<SplitVisitor> {
public:
    SplitVisitor(ASTContext &context, SplitOutput &output,
                 const std::string &input_name)
        : context_(context), source_(context.getSourceManager()),
          language_(context.getLangOpts()), output_(output),
          input_name_(input_name), policy_(context.getLangOpts())
    {
        policy_.SuppressTagKeyword = false;
        policy_.FullyQualifiedName = true;
        policy_.PrintAsCanonical = false;
    }

    bool VisitFunctionDecl(FunctionDecl *decl)
    {
        if (!is_main_file_decl(decl) || decl->isImplicit() ||
            !decl->getDeclContext()->getRedeclContext()->isFileContext())
            return true;
        functions_.push_back(decl);
        return true;
    }

    bool VisitVarDecl(VarDecl *decl)
    {
        if (!is_main_file_decl(decl) || decl->isImplicit() ||
            !decl->isFileVarDecl())
            return true;
        variables_.push_back(decl);
        return true;
    }

    void finish()
    {
        collect_function_edits();
        collect_variable_edits();
        if (output_.failed)
            return;

        std::string original = source_.getBufferData(source_.getMainFileID()).str();
        output_.host = apply_edits(original, host_edits_, "Host");
        output_.device = apply_edits(original, device_edits_, "Device");
        if (output_.failed)
            return;

        std::ostringstream prelude;
        prelude << "/* Generated by dycc. */\n"
                << "#include <dayan_kernel.h>\n"
                << "extern \"C\" {\n"
                << "extern const unsigned char __kernel_obj__[];\n"
                << "extern const size_t __kernel_obj_size__;\n"
                << "}\n"
                << "#line 1 \"" << escape_string(input_name_) << "\"\n";
        output_.host.insert(0, prelude.str());
        output_.device.insert(0, "/* Generated by dycc. */\n#line 1 \"" +
                                     escape_string(input_name_) + "\"\n");
    }

private:
    struct Span {
        std::size_t begin;
        std::size_t end;
    };

    bool is_main_file_decl(const Decl *decl) const
    {
        SourceLocation location = source_.getExpansionLoc(decl->getBeginLoc());
        return location.isValid() && source_.isWrittenInMainFile(location);
    }

    void diagnose(const Decl *decl, const std::string &message)
    {
        unsigned id = context_.getDiagnostics().getCustomDiagID(
            DiagnosticsEngine::Error, "%0");
        context_.getDiagnostics().Report(decl->getLocation(), id) << message;
        output_.failed = true;
    }

    std::optional<Span> declaration_span(const Decl *decl,
                                         bool consume_semicolon)
    {
        const Decl *range_decl = decl;
        if (const auto *linkage = llvm::dyn_cast<clang::LinkageSpecDecl>(
                decl->getLexicalDeclContext())) {
            if (!linkage->hasBraces())
                range_decl = linkage;
        }
        SourceLocation begin = source_.getExpansionLoc(range_decl->getBeginLoc());
        SourceLocation end = source_.getExpansionLoc(decl->getEndLoc());
        if (begin.isInvalid() || end.isInvalid() ||
            !source_.isWrittenInMainFile(begin) ||
            !source_.isWrittenInMainFile(end)) {
            diagnose(decl, "dycc cannot rewrite a declaration produced by a macro");
            return std::nullopt;
        }
        SourceLocation after = Lexer::getLocForEndOfToken(end, 0, source_, language_);
        if (after.isInvalid()) {
            diagnose(decl, "dycc cannot determine the end of this declaration");
            return std::nullopt;
        }
        if (consume_semicolon) {
            auto next = Lexer::findNextToken(end, source_, language_);
            if (next && next->is(clang::tok::semi)) {
                after = Lexer::getLocForEndOfToken(next->getLocation(), 0,
                                                    source_, language_);
            }
        }
        std::size_t begin_offset = source_.getFileOffset(begin);
        std::size_t end_offset = source_.getFileOffset(after);
        if (begin_offset >= end_offset) {
            diagnose(decl, "dycc computed an invalid declaration source range");
            return std::nullopt;
        }
        return Span{begin_offset, end_offset};
    }

    std::optional<Span> variable_statement_span(const VarDecl *decl)
    {
        auto span = declaration_span(decl, false);
        unsigned brace_depth = 0;
        unsigned bracket_depth = 0;
        unsigned parenthesis_depth = 0;

        if (!span)
            return std::nullopt;
        SourceLocation cursor = source_.getExpansionLoc(decl->getEndLoc());
        for (;;) {
            auto next = Lexer::findNextToken(cursor, source_, language_);
            if (!next) {
                diagnose(decl, "dycc cannot find ';' after global variable declaration");
                return std::nullopt;
            }
            cursor = next->getLocation();
            if (next->is(clang::tok::semi) && !brace_depth &&
                !bracket_depth && !parenthesis_depth) {
                SourceLocation after = Lexer::getLocForEndOfToken(
                    cursor, 0, source_, language_);
                span->end = source_.getFileOffset(after);
                return span;
            }
            if (next->is(clang::tok::l_brace))
                ++brace_depth;
            else if (next->is(clang::tok::l_square))
                ++bracket_depth;
            else if (next->is(clang::tok::l_paren))
                ++parenthesis_depth;
            else if (next->is(clang::tok::r_brace) && brace_depth)
                --brace_depth;
            else if (next->is(clang::tok::r_square) && bracket_depth)
                --bracket_depth;
            else if (next->is(clang::tok::r_paren) && parenthesis_depth)
                --parenthesis_depth;
            else if (next->is(clang::tok::r_brace) ||
                     next->is(clang::tok::r_square) ||
                     next->is(clang::tok::r_paren)) {
                diagnose(decl, "dycc cannot isolate this global variable declaration");
                return std::nullopt;
            }
        }
    }

    std::optional<Span> skipped_function_span(const FunctionDecl *decl)
    {
        auto span = declaration_span(decl, false);
        SourceLocation cursor = source_.getExpansionLoc(decl->getEndLoc());
        unsigned depth = 0;

        if (!span)
            return std::nullopt;
        for (;;) {
            auto next = Lexer::findNextToken(cursor, source_, language_);
            if (!next) {
                diagnose(decl, "dycc cannot find the skipped function body");
                return std::nullopt;
            }
            cursor = next->getLocation();
            if (!depth) {
                if (next->is(clang::tok::semi) ||
                    next->is(clang::tok::r_brace)) {
                    diagnose(decl, "dycc cannot find the skipped function body");
                    return std::nullopt;
                }
                if (next->is(clang::tok::l_brace))
                    depth = 1;
                continue;
            }
            if (next->is(clang::tok::l_brace)) {
                ++depth;
            } else if (next->is(clang::tok::r_brace) && --depth == 0) {
                SourceLocation after = Lexer::getLocForEndOfToken(
                    cursor, 0, source_, language_);
                span->end = source_.getFileOffset(after);
                return span;
            }
        }
    }

    static std::string escape_string(const std::string &value)
    {
        std::string escaped;
        for (char c : value) {
            if (c == '\\' || c == '"')
                escaped.push_back('\\');
            escaped.push_back(c);
        }
        return escaped;
    }

    std::string source_text(SourceRange range) const
    {
        if (range.isInvalid())
            return {};
        return Lexer::getSourceText(CharSourceRange::getTokenRange(range),
                                    source_, language_).str();
    }

    std::string parameter_declaration(const ParmVarDecl *parameter,
                                      unsigned index, bool with_default)
    {
        std::string name = parameter->getNameAsString();
        if (name.empty())
            name = "__dayan_arg_" + std::to_string(index);
        QualType canonical = parameter->getType().getCanonicalType();
        std::string rendered;
        if (canonical->isReferenceType() || canonical->isPointerType()) {
            rendered = "::npu_buffer_t *" + name;
            return rendered;
        }
        llvm::raw_string_ostream stream(rendered);
        parameter->getOriginalType().print(stream, policy_, name);
        stream.flush();
        if (with_default && parameter->hasDefaultArg() &&
            !parameter->hasInheritedDefaultArg()) {
            std::string expression = source_text(parameter->getDefaultArgRange());
            if (!expression.empty())
                rendered += " = " + expression;
        }
        return rendered;
    }

    std::string argument_expression(const ParmVarDecl *parameter,
                                    unsigned index)
    {
        std::string name = parameter->getNameAsString();
        if (name.empty())
            name = "__dayan_arg_" + std::to_string(index);
        QualType type = parameter->getType();
        QualType canonical = type.getCanonicalType();
        if (canonical->isReferenceType() || canonical->isPointerType())
            return name;
        if (canonical->isNullPtrType())
            return "UINT64_C(0)";
        if (canonical->isIntegralOrEnumerationType())
            return "static_cast<uint64_t>(" + name + ")";
        if (canonical->isSpecificBuiltinType(clang::BuiltinType::Float))
            return "::npu_kernel_pack_f32(" + name + ")";
        if (canonical->isSpecificBuiltinType(clang::BuiltinType::Double))
            return "::npu_kernel_pack_f64(" + name + ")";
        return {};
    }

    bool buffer_parameter(const ParmVarDecl *parameter) const
    {
        QualType canonical = parameter->getType().getCanonicalType();
        return canonical->isReferenceType() || canonical->isPointerType();
    }

    uint64_t argument_size(const ParmVarDecl *parameter) const
    {
        QualType canonical = parameter->getType().getCanonicalType();
        if (buffer_parameter(parameter) || canonical->isNullPtrType())
            return 8;
        return static_cast<uint64_t>(
            context_.getTypeSizeInChars(canonical).getQuantity());
    }

    uint64_t buffer_alignment(const ParmVarDecl *parameter) const
    {
        QualType canonical = parameter->getType().getCanonicalType();
        QualType pointee = canonical->getPointeeType();
        if (pointee.isNull() || pointee->isVoidType() ||
            pointee->isFunctionType() || pointee->isIncompleteType())
            return 1;
        return static_cast<uint64_t>(
            context_.getTypeAlignInChars(pointee).getQuantity());
    }

    uint32_t buffer_access(const ParmVarDecl *parameter) const
    {
        QualType pointee =
            parameter->getType().getCanonicalType()->getPointeeType();
        return pointee.isConstQualified() ? 1U : 3U;
    }

    std::string mangled_name(const FunctionDecl *decl)
    {
        std::unique_ptr<MangleContext> mangler(context_.createMangleContext());
        if (!mangler->shouldMangleDeclName(decl))
            return decl->getNameAsString();
        std::string result;
        llvm::raw_string_ostream stream(result);
        mangler->mangleName(GlobalDecl(decl), stream);
        stream.flush();
        return result;
    }

    std::string wrapper(const FunctionDecl *decl, bool definition)
    {
        std::ostringstream out;
        unsigned line = source_.getSpellingLineNumber(
            source_.getExpansionLoc(decl->getBeginLoc()));
        unsigned after_line = source_.getSpellingLineNumber(
            source_.getExpansionLoc(decl->getEndLoc())) + 1;
        const auto *linkage = llvm::dyn_cast<clang::LinkageSpecDecl>(
            decl->getLexicalDeclContext());
        out << "#line " << line << " \"" << escape_string(input_name_) << "\"\n";
        if (linkage && !linkage->hasBraces())
            out << "extern \"C\" ";
        out << "int " << decl->getNameAsString() << "(";
        for (unsigned i = 0; i < decl->getNumParams(); ++i) {
            if (i)
                out << ", ";
            out << parameter_declaration(decl->getParamDecl(i), i, false);
        }
        if (decl->getNumParams())
            out << ", ";
        out << "const ::npu_kernel_launch_attrs_t *__dayan_launch_attrs)";
        if (!definition) {
            out << ";\n#line " << after_line << " \""
                << escape_string(input_name_) << "\"\n";
            return out.str();
        }

        out << "\n{\n"
            << "    static ::npu_kernel_descriptor_t *__dayan_descriptor = nullptr;\n"
            << "    static unsigned char __dayan_init_lock = 0;\n";
        if (decl->getNumParams()) {
            out << "    static const ::npu_package_argument_t __dayan_metadata[] = {\n";
            for (unsigned i = 0; i < decl->getNumParams(); ++i) {
                const ParmVarDecl *parameter = decl->getParamDecl(i);
                if (buffer_parameter(parameter)) {
                    out << "        { ::NPU_PACKAGE_ARG_BUFFER, 8U, "
                        << buffer_alignment(parameter) << "U, "
                        << buffer_access(parameter) << "U }";
                } else {
                    out << "        { ::NPU_PACKAGE_ARG_VALUE, "
                        << argument_size(parameter) << "U, 0U, 0U }";
                }
                out << (i + 1 == decl->getNumParams() ? "\n" : ",\n");
            }
            out << "    };\n";
        }
        out
            << "    ::npu_kernel_descriptor_t *__dayan_local =\n"
            << "        __atomic_load_n(&__dayan_descriptor, __ATOMIC_ACQUIRE);\n"
            << "    if (!__dayan_local) {\n"
            << "        while (__atomic_test_and_set(&__dayan_init_lock, "
               "__ATOMIC_ACQUIRE))\n"
            << "            ;\n"
            << "        __dayan_local = __atomic_load_n(&__dayan_descriptor, "
               "__ATOMIC_RELAXED);\n"
            << "        if (!__dayan_local) {\n"
            << "            int __dayan_status = ::npu_kernel_descriptor_create(\n"
            << "                ::__kernel_obj__, ::__kernel_obj_size__, \""
            << escape_string(mangled_name(decl)) << "\", ";
        if (decl->getNumParams())
            out << "__dayan_metadata, " << decl->getNumParams();
        else
            out << "nullptr, 0";
        out << ", &__dayan_local);\n"
            << "            if (__dayan_status < 0) {\n"
            << "                __atomic_clear(&__dayan_init_lock, __ATOMIC_RELEASE);\n"
            << "                return __dayan_status;\n"
            << "            }\n"
            << "            __atomic_store_n(&__dayan_descriptor, __dayan_local, "
               "__ATOMIC_RELEASE);\n"
            << "        }\n"
            << "        __atomic_clear(&__dayan_init_lock, __ATOMIC_RELEASE);\n"
            << "    }\n";
        if (decl->getNumParams()) {
            out << "    ::npu_kernel_arg_t __dayan_args["
                << decl->getNumParams() << "] = {};\n";
            for (unsigned i = 0; i < decl->getNumParams(); ++i) {
                const ParmVarDecl *parameter = decl->getParamDecl(i);
                if (buffer_parameter(parameter)) {
                    out << "    ::npu_kernel_arg_buffer(&__dayan_args[" << i
                        << "], " << argument_expression(parameter, i)
                        << ", 0, " << buffer_access(parameter) << "U);\n";
                } else {
                    out << "    ::npu_kernel_arg_value(&__dayan_args[" << i
                        << "], " << argument_expression(parameter, i)
                        << ", " << argument_size(parameter) << "U);\n";
                }
            }
            out << "    return ::npu_kernel_descriptor_launch(__dayan_local, __dayan_args, "
                << decl->getNumParams() << ", __dayan_launch_attrs);\n";
        } else {
            out << "    return ::npu_kernel_descriptor_launch(__dayan_local, nullptr, 0, "
                   "__dayan_launch_attrs);\n";
        }
        out << "}\n#line " << after_line << " \""
            << escape_string(input_name_) << "\"\n";
        return out.str();
    }

    bool validate_kernel(FunctionDecl *decl)
    {
        if (!decl->getIdentifier()) {
            diagnose(decl, "__kernel__ requires an ordinary named function");
            return false;
        }
        if (decl->getTemplatedKind() != FunctionDecl::TK_NonTemplate) {
            diagnose(decl, "__kernel__ function templates are not supported");
            return false;
        }
        if (decl->isVariadic()) {
            diagnose(decl, "__kernel__ functions cannot be variadic");
            return false;
        }
        if (!decl->getReturnType()->isVoidType()) {
            diagnose(decl, "__kernel__ functions must return void");
            return false;
        }
        if (decl->getNumParams() > 8) {
            diagnose(decl, "__kernel__ functions support at most 8 machine-word arguments");
            return false;
        }
        if (decl->getFormalLinkage() == clang::Linkage::Internal) {
            diagnose(decl, "__kernel__ functions must have external linkage");
            return false;
        }
        for (unsigned i = 0; i < decl->getNumParams(); ++i) {
            if (argument_expression(decl->getParamDecl(i), i).empty()) {
                diagnose(decl->getParamDecl(i),
                         "kernel parameter must be an integer, enum, pointer, reference, float, or double");
                return false;
            }
        }
        return true;
    }

    void collect_function_edits()
    {
        std::set<const FunctionDecl *> validated;
        std::map<const FunctionDecl *, unsigned> definitions;
        for (FunctionDecl *decl : functions_) {
            const FunctionDecl *canonical = decl->getCanonicalDecl();
            Side side = function_side(decl);
            if (side == Side::Invalid) {
                diagnose(decl, "function redeclarations have conflicting Dayan attributes");
                continue;
            }
            if (side == Side::Kernel && validated.insert(canonical).second) {
                if (!validate_kernel(decl))
                    continue;
                for (const FunctionDecl *redecl : decl->redecls()) {
                    if (redecl->isThisDeclarationADefinition())
                        ++definitions[canonical];
                }
                if (definitions[canonical] != 1) {
                    diagnose(decl, "__kernel__ requires exactly one definition in this source file");
                    continue;
                }
            }

            bool has_body = decl->isThisDeclarationADefinition();
            auto span = decl->hasSkippedBody()
                            ? skipped_function_span(decl)
                            : declaration_span(decl, !has_body);
            if (!span)
                continue;
            switch (side) {
            case Side::Both:
                break;
            case Side::Host:
                device_edits_.push_back({span->begin, span->end, {}});
                break;
            case Side::Device:
                host_edits_.push_back({span->begin, span->end, {}});
                break;
            case Side::Kernel:
                host_edits_.push_back(
                    {span->begin, span->end, wrapper(decl, has_body)});
                break;
            case Side::Invalid:
                break;
            }
        }
    }

    void collect_variable_edits()
    {
        struct Group {
            std::size_t begin = std::numeric_limits<std::size_t>::max();
            std::size_t end = 0;
            Side side = Side::Both;
            VarDecl *representative = nullptr;
        };
        std::map<std::size_t, Group> groups;

        for (VarDecl *decl : variables_) {
            Side side = annotation_side(decl);
            if (side == Side::Both) {
                diagnose(decl, "global variable requires an explicit __host__ or __device__ attribute");
                continue;
            }
            if (side == Side::Kernel || side == Side::Invalid) {
                diagnose(decl, "global variable must use exactly one of __host__ or __device__");
                continue;
            }
            auto span = variable_statement_span(decl);
            if (!span)
                continue;
            Group &group = groups[span->end];
            group.begin = std::min(group.begin, span->begin);
            group.end = span->end;
            group.representative = decl;
            if (group.side == Side::Both)
                group.side = side;
            else if (group.side != side)
                group.side = Side::Invalid;
        }
        for (const auto &[unused, group] : groups) {
            (void)unused;
            if (group.side == Side::Invalid) {
                diagnose(group.representative,
                         "variables in one declaration cannot target different Dayan sides");
                continue;
            }
            Edit edit{group.begin, group.end, {}};
            if (group.side == Side::Host)
                device_edits_.push_back(edit);
            else if (group.side == Side::Device)
                host_edits_.push_back(edit);
        }
    }

    std::string apply_edits(const std::string &original,
                            std::vector<Edit> edits,
                            const char *side_name)
    {
        std::sort(edits.begin(), edits.end(), [](const Edit &left,
                                                 const Edit &right) {
            if (left.begin != right.begin)
                return left.begin > right.begin;
            return left.end > right.end;
        });
        std::string result = original;
        std::size_t previous_begin = original.size();
        for (const Edit &edit : edits) {
            if (edit.end > original.size() || edit.begin >= edit.end ||
                edit.end > previous_begin) {
                std::cerr << "dycc: error: overlapping " << side_name
                          << " source edits near byte " << edit.begin << "\n";
                output_.failed = true;
                return {};
            }
            result.replace(edit.begin, edit.end - edit.begin, edit.replacement);
            previous_begin = edit.begin;
        }
        return result;
    }

    ASTContext &context_;
    SourceManager &source_;
    const LangOptions &language_;
    SplitOutput &output_;
    std::string input_name_;
    PrintingPolicy policy_;
    std::vector<FunctionDecl *> functions_;
    std::vector<VarDecl *> variables_;
    std::vector<Edit> host_edits_;
    std::vector<Edit> device_edits_;
};

class SplitConsumer : public clang::ASTConsumer {
public:
    SplitConsumer(ASTContext &context, SplitOutput &output,
                  const std::string &input_name)
        : visitor_(context, output, input_name)
    {
    }

    void HandleTranslationUnit(ASTContext &context) override
    {
        visitor_.TraverseDecl(context.getTranslationUnitDecl());
        visitor_.finish();
    }

    bool shouldSkipFunctionBody(Decl *decl) override
    {
        const auto *function = llvm::dyn_cast<FunctionDecl>(decl);

        return function && function_side(function) == Side::Host;
    }

private:
    SplitVisitor visitor_;
};

class SplitAction : public clang::ASTFrontendAction {
public:
    SplitAction(SplitOutput &output, std::string input_name)
        : output_(output), input_name_(std::move(input_name))
    {
    }

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(CompilerInstance &compiler, llvm::StringRef) override
    {
        return std::make_unique<SplitConsumer>(compiler.getASTContext(), output_,
                                               input_name_);
    }

    bool BeginSourceFileAction(CompilerInstance &compiler) override
    {
        compiler.getFrontendOpts().SkipFunctionBodies = true;
        return true;
    }

private:
    SplitOutput &output_;
    std::string input_name_;
};

class SplitActionFactory : public clang::tooling::FrontendActionFactory {
public:
    SplitActionFactory(SplitOutput &output, std::string input_name)
        : output_(output), input_name_(std::move(input_name))
    {
    }

    std::unique_ptr<clang::FrontendAction> create() override
    {
        return std::make_unique<SplitAction>(output_, input_name_);
    }

private:
    SplitOutput &output_;
    std::string input_name_;
};

static bool split_source(const Options &options, SplitOutput &output)
{
    std::error_code error;
    fs::path absolute = fs::absolute(options.input, error);
    if (error) {
        std::cerr << "dycc: error: cannot resolve input path: "
                  << error.message() << "\n";
        return false;
    }
    std::vector<std::string> arguments = {
        "-x", "c++", "-std=c++17", "-fsyntax-only",
        "--target=riscv64-unknown-elf", "-march=rv64imaf_zicsr_zifencei",
        "-mabi=lp64", "-D__DAYAN_DEVICE__=1", "-include",
        (options.root / "host/dycc/include/dayan_compiler.h").string(),
        "-isystem", (options.root / "include/api").string(),
        "-isystem", (options.root / "host/include").string(),
    };
    arguments.insert(arguments.end(), options.common_args.begin(),
                     options.common_args.end());
    arguments.insert(arguments.end(), options.device_args.begin(),
                     options.device_args.end());

    clang::tooling::FixedCompilationDatabase database(
        fs::current_path().string(), arguments);
    clang::tooling::ClangTool tool(database, {absolute.string()});
    SplitActionFactory factory(output, absolute.string());
    int status = tool.run(&factory);
    return status == 0 && !output.failed;
}

static bool write_file(const fs::path &path, const std::string &contents)
{
    std::error_code error;
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path(), error);
    if (error) {
        std::cerr << "dycc: error: cannot create directory for '" << path
                  << "': " << error.message() << "\n";
        return false;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        std::cerr << "dycc: error: cannot write '" << path << "'\n";
        return false;
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream) {
        std::cerr << "dycc: error: failed while writing '" << path << "'\n";
        return false;
    }
    return true;
}

static std::string quote_command_arg(const std::string &arg)
{
    if (arg.find_first_of(" \t\n'\"\\$`") == std::string::npos)
        return arg;
    std::string result = "'";
    for (char c : arg) {
        if (c == '\'')
            result += "'\\''";
        else
            result.push_back(c);
    }
    result.push_back('\'');
    return result;
}

static std::optional<std::string> find_program(const fs::path &program)
{
    if (program.has_parent_path()) {
        std::error_code error;
        fs::path absolute = fs::absolute(program, error);
        if (!error && fs::exists(absolute))
            return absolute.string();
        return std::nullopt;
    }
    auto found = llvm::sys::findProgramByName(program.string());
    if (!found)
        return std::nullopt;
    return *found;
}

static int run_command(const fs::path &program,
                       const std::vector<std::string> &arguments,
                       const Options &options)
{
    auto executable = find_program(program);
    if (!executable) {
        std::cerr << "dycc: error: cannot find executable '"
                  << program.string() << "'\n";
        return 1;
    }
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.push_back(*executable);
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    if (options.verbose || options.dry_run) {
        std::cerr << "dycc:";
        for (const std::string &arg : storage)
            std::cerr << ' ' << quote_command_arg(arg);
        std::cerr << "\n";
    }
    if (options.dry_run)
        return 0;
    std::vector<llvm::StringRef> refs;
    refs.reserve(storage.size());
    for (const std::string &arg : storage)
        refs.emplace_back(arg);
    std::string error;
    bool failed = false;
    int status = llvm::sys::ExecuteAndWait(*executable, refs, std::nullopt, {},
                                           0, 0, &error, &failed);
    if (failed || status < 0) {
        std::cerr << "dycc: error: failed to execute '" << program.string()
                  << "'";
        if (!error.empty())
            std::cerr << ": " << error;
        std::cerr << "\n";
        return 1;
    }
    return status;
}

static bool read_binary(const fs::path &path, std::vector<unsigned char> &data)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::cerr << "dycc: error: cannot read '" << path << "'\n";
        return false;
    }
    stream.seekg(0, std::ios::end);
    std::streamoff length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (length <= 0 || static_cast<std::uintmax_t>(length) >
                           std::numeric_limits<std::size_t>::max()) {
        std::cerr << "dycc: error: invalid Device object size\n";
        return false;
    }
    data.resize(static_cast<std::size_t>(length));
    stream.read(reinterpret_cast<char *>(data.data()), length);
    return static_cast<bool>(stream);
}

static std::string embedded_source(const std::vector<unsigned char> &data)
{
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(data.size() * 6 + 512);
    result += "#include <stddef.h>\n"
              "extern \"C\" {\n"
              "__attribute__((aligned(16), visibility(\"hidden\"), "
              "section(\".rodata.dayan.kernel\")))\n"
              "extern const unsigned char __kernel_obj__[] = {\n";
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i % 12 == 0)
            result += "    ";
        result += "0x";
        result.push_back(hex[data[i] >> 4]);
        result.push_back(hex[data[i] & 15]);
        result += i + 1 == data.size() ? "\n" : ", ";
        if (i % 12 == 11 && i + 1 != data.size())
            result += "\n";
    }
    result += "};\n"
              "__attribute__((visibility(\"hidden\")))\n"
              "extern const size_t __kernel_obj_size__ = sizeof(__kernel_obj__);\n"
              "}\n";
    return result;
}

static fs::path make_temp_directory()
{
    fs::path base;
    if (const char *temporary = std::getenv("TMPDIR"))
        base = temporary;
    else
        base = "/tmp";
    std::string pattern = (base / "dycc-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char *created = ::mkdtemp(buffer.data());
    return created ? fs::path(created) : fs::path();
}

static std::vector<std::string> device_compile_args(
    const Options &options, const fs::path &source, const fs::path &object)
{
    std::vector<std::string> args = {
        "-x", "c++", "-std=c++17", "--target=riscv64-unknown-elf",
        "-march=rv64imaf_zicsr_zifencei", "-mabi=lp64", "-mcmodel=medany",
        "-mno-relax", "-msmall-data-limit=0", "-ffreestanding",
        "-fno-builtin", "-fno-common", "-fno-pic", "-fno-pie",
        "-fno-stack-protector", "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables", "-fno-exceptions", "-fno-rtti",
        "-fno-threadsafe-statics", "-fno-use-cxa-atexit",
        "-ffunction-sections", "-fdata-sections", "-O2",
        "-D__DAYAN_DEVICE__=1", "-D__host__=",
        "-D__device__=__attribute__((used))",
        "-D__kernel__=__attribute__((used,visibility(\"default\")))",
        "-isystem", (options.root / "include/api").string(),
        "-isystem", (options.root / "host/include").string(),
    };
    args.insert(args.end(), options.common_args.begin(), options.common_args.end());
    args.insert(args.end(), options.device_args.begin(), options.device_args.end());
    args.insert(args.end(), {"-c", source.string(), "-o", object.string()});
    return args;
}

static std::vector<std::string> host_compile_args(
    const Options &options, const fs::path &source, const fs::path &object)
{
    std::vector<std::string> args = {
        "-x", "c++", "-std=c++17", "-O2", "-D__DAYAN_HOST__=1",
        "-D__host__=", "-D__device__=", "-D__kernel__=",
        "-isystem", (options.root / "host/include").string(),
    };
    args.insert(args.end(), options.common_args.begin(), options.common_args.end());
    args.insert(args.end(), options.host_args.begin(), options.host_args.end());
    args.insert(args.end(), {"-c", source.string(), "-o", object.string()});
    return args;
}

static fs::path temp_path(const Options &options, const fs::path &work,
                          const std::string &suffix)
{
    std::string stem = options.input.filename().string();
    return work / (stem + suffix);
}

static int compile_object(const Options &options)
{
    if (!fs::is_regular_file(options.input)) {
        std::cerr << "dycc: error: input is not a regular file: '"
                  << options.input << "'\n";
        return 1;
    }
    if (!is_dayan_root(options.root)) {
        std::cerr << "dycc: error: invalid Dayan root '" << options.root
                  << "'\n";
        return 1;
    }

    SplitOutput split;
    if (!split_source(options, split))
        return 1;

    fs::path temporary;
    fs::path work;
    if (options.save_temps) {
        work = options.save_temps_dir.empty()
                   ? fs::absolute(options.output.empty() ? fs::path(".")
                                                         : options.output)
                         .parent_path()
                   : options.save_temps_dir;
        std::error_code error;
        fs::create_directories(work, error);
        if (error) {
            std::cerr << "dycc: error: cannot create temporary output directory: "
                      << error.message() << "\n";
            return 1;
        }
    } else {
        temporary = make_temp_directory();
        if (temporary.empty()) {
            std::cerr << "dycc: error: cannot create temporary directory: "
                      << std::strerror(errno) << "\n";
            return 1;
        }
        work = temporary;
    }

    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            if (!path.empty()) {
                std::error_code ignored;
                fs::remove_all(path, ignored);
            }
        }
    } cleanup{temporary};

    fs::path host_source = temp_path(options, work, ".dycc.host.cpp");
    fs::path device_source = temp_path(options, work, ".dycc.device.cpp");
    fs::path host_object = temp_path(options, work, ".dycc.host.o");
    fs::path device_object = temp_path(options, work, ".dycc.device.o");
    fs::path embed_source = temp_path(options, work, ".dycc.embed.cpp");
    fs::path embed_object = temp_path(options, work, ".dycc.embed.o");

    if (!write_file(host_source, split.host) ||
        !write_file(device_source, split.device))
        return 1;
    if (!options.emit_host.empty() && !write_file(options.emit_host, split.host))
        return 1;
    if (!options.emit_device.empty() &&
        !write_file(options.emit_device, split.device))
        return 1;
    if (options.split_only)
        return 0;

    if (run_command(options.device_cxx,
                    device_compile_args(options, device_source, device_object),
                    options) != 0)
        return 1;
    if (options.dry_run) {
        std::cerr << "dycc: note: embed and final-link commands require the "
                     "Device object and are omitted in --dry-run mode\n";
        return 0;
    }

    std::vector<unsigned char> image;
    if (!read_binary(device_object, image) ||
        !write_file(embed_source, embedded_source(image)))
        return 1;
    if (run_command(options.host_cxx,
                    host_compile_args(options, host_source, host_object),
                    options) != 0)
        return 1;
    if (run_command(options.host_cxx,
                    host_compile_args(options, embed_source, embed_object),
                    options) != 0)
        return 1;

    std::error_code error;
    if (!options.output.parent_path().empty())
        fs::create_directories(options.output.parent_path(), error);
    if (error) {
        std::cerr << "dycc: error: cannot create output directory: "
                  << error.message() << "\n";
        return 1;
    }
    std::vector<std::string> link_args = {
        "-r", "-nostdlib", host_object.string(), embed_object.string(),
        "-o", options.output.string(),
    };
    if (run_command(options.host_cxx, link_args, options) != 0)
        return 1;
    std::vector<std::string> localize_args = {
        "--localize-symbol=__kernel_obj__",
        "--localize-symbol=__kernel_obj_size__", options.output.string(),
    };
    if (run_command(options.objcopy, localize_args, options) != 0)
        return 1;
    return 0;
}

static std::optional<fs::path> find_runtime_archive(const Options &options)
{
    if (const char *override_path = std::getenv("DYCC_RUNTIME_LIBRARY")) {
        fs::path path = override_path;
        if (fs::is_regular_file(path))
            return path;
        std::cerr << "dycc: error: DYCC_RUNTIME_LIBRARY is not a regular "
                     "file: '"
                  << path << "'\n";
        return std::nullopt;
    }

    fs::path prefix = options.root.parent_path().parent_path();
    const fs::path candidates[] = {
        options.root / "build/host/driver/libDayanRuntime.a",
        options.root.parent_path() / "libDayanRuntime.a",
        prefix / "lib/libDayanRuntime.a",
        prefix / "lib64/libDayanRuntime.a",
    };
    for (const fs::path &candidate : candidates) {
        if (fs::is_regular_file(candidate))
            return candidate;
    }
    std::cerr << "dycc: error: cannot find libDayanRuntime.a; build the Host "
                 "runtime or set DYCC_RUNTIME_LIBRARY\n";
    return std::nullopt;
}

static int link_executable(const Options &options,
                           const std::vector<fs::path> &objects)
{
    auto runtime = find_runtime_archive(options);
    if (!runtime)
        return 1;

    std::error_code error;
    if (!options.output.parent_path().empty())
        fs::create_directories(options.output.parent_path(), error);
    if (error) {
        std::cerr << "dycc: error: cannot create output directory: "
                  << error.message() << "\n";
        return 1;
    }

    std::vector<std::string> arguments;
    arguments.insert(arguments.end(), options.common_args.begin(),
                     options.common_args.end());
    arguments.insert(arguments.end(), options.host_args.begin(),
                     options.host_args.end());
    for (const fs::path &object : objects)
        arguments.push_back(object.string());
    arguments.insert(arguments.end(), options.link_args.begin(),
                     options.link_args.end());
    arguments.insert(arguments.end(), {
        runtime->string(), "-pthread", "-o", options.output.string(),
    });
    return run_command(options.host_cxx, arguments, options);
}

static int run_driver(const Options &options)
{
    if (options.split_only) {
        Options unit = options;
        unit.input = options.source_inputs.front();
        return compile_object(unit);
    }

    if (options.compile_only) {
        for (const fs::path &source : options.source_inputs) {
            Options unit = options;
            unit.input = source;
            if (options.source_inputs.size() == 1) {
                unit.output = options.output;
            } else {
                unit.output = source.filename();
                unit.output.replace_extension(".o");
            }
            if (compile_object(unit) != 0)
                return 1;
        }
        return 0;
    }

    fs::path temporary;
    fs::path work;
    if (options.save_temps) {
        work = options.save_temps_dir.empty()
                   ? fs::absolute(options.output).parent_path()
                   : options.save_temps_dir;
        std::error_code error;
        fs::create_directories(work, error);
        if (error) {
            std::cerr << "dycc: error: cannot create link work directory: "
                      << error.message() << "\n";
            return 1;
        }
    } else {
        temporary = make_temp_directory();
        if (temporary.empty()) {
            std::cerr << "dycc: error: cannot create temporary directory: "
                      << std::strerror(errno) << "\n";
            return 1;
        }
        work = temporary;
    }

    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            if (!path.empty()) {
                std::error_code ignored;
                fs::remove_all(path, ignored);
            }
        }
    } cleanup{temporary};

    std::vector<fs::path> objects;
    objects.reserve(options.source_inputs.size());
    for (std::size_t index = 0; index < options.source_inputs.size(); ++index) {
        const fs::path &source = options.source_inputs[index];
        Options unit = options;
        unit.input = source;
        unit.output = work / (source.filename().string() + "." +
                              std::to_string(index) + ".dycc.o");
        if (compile_object(unit) != 0)
            return 1;
        objects.push_back(unit.output);
    }
    return link_executable(options, objects);
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(std::cerr);
        return 1;
    }
    return run_driver(options);
}
