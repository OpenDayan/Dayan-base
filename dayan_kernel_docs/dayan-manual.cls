\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dayan-manual}[2026/08/02 Dayan kernel manual]

\LoadClass[UTF8,zihao=5,fontset=none,a4paper,oneside]{ctexart}

\RequirePackage{fontspec}
\RequirePackage{geometry}
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{tabularray}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{fancyhdr}
\PassOptionsToPackage{obeyspaces,spaces}{url}
\RequirePackage{hyperref}
\RequirePackage{bookmark}
\RequirePackage{tcolorbox}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\RequirePackage{fvextra}
\tcbuselibrary{skins,breakable}

\definecolor{DayanText}{HTML}{262626}
\definecolor{DayanNavy}{HTML}{1D3557}
\definecolor{DayanSecondary}{HTML}{4A5568}
\definecolor{DayanLink}{HTML}{1D5F99}
\definecolor{DayanCode}{HTML}{17324D}
\definecolor{DayanTableBorder}{HTML}{D5DAE0}
\definecolor{DayanTableHead}{HTML}{EDF2F7}
\definecolor{DayanTableStripe}{HTML}{F8FAFC}
\definecolor{DayanRule}{HTML}{8DA7BE}
\definecolor{DayanMuted}{HTML}{718096}

\defaultfontfeatures{Ligatures=TeX}
\IfFontExistsTF{Noto Sans CJK SC}{
  \setmainfont{Noto Sans CJK SC}
  \setsansfont{Noto Sans CJK SC}
  \setCJKmainfont{Noto Sans CJK SC}
  \setCJKsansfont{Noto Sans CJK SC}
  \setCJKmonofont{Noto Sans CJK SC}
}{
  \setmainfont{FandolSong}
  \setsansfont{FandolHei}
  \setCJKmainfont{FandolSong}
  \setCJKsansfont{FandolHei}
  \setCJKmonofont{FandolFang}
}
\IfFontExistsTF{DejaVu Sans Mono}{
  \setmonofont{DejaVu Sans Mono}
}{
  \setmonofont{Latin Modern Mono}
}

\geometry{
  top=0.92in,
  right=0.88in,
  bottom=0.86in,
  left=0.94in,
  headheight=14pt,
  headsep=8pt,
  footskip=22pt
}

\color{DayanText}
\AtBeginDocument{\fontsize{10.5pt}{16.2pt}\selectfont}
\setlength{\parindent}{21pt}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\raggedbottom

\setlist[itemize]{
  leftmargin=2em,
  itemsep=2pt,
  topsep=4pt,
  parsep=0pt,
  partopsep=0pt
}
\setlist[enumerate]{
  leftmargin=2.2em,
  itemsep=2pt,
  topsep=4pt,
  parsep=0pt,
  partopsep=0pt
}

\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
\renewcommand{\theparagraph}{\thesubsubsection.\arabic{paragraph}}
\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{3}

\ctexset{
  section={
    name={},
    number=\thesection,
    format=\color{DayanNavy}\bfseries\fontsize{20pt}{24pt}\selectfont,
    beforeskip=8pt,
    afterskip=8pt
  },
  subsection={
    name={},
    number=\thesubsection,
    format=\color{DayanNavy}\bfseries\fontsize{16pt}{19.2pt}\selectfont,
    beforeskip=8pt,
    afterskip=8pt
  },
  subsubsection={
    name={},
    number=\thesubsubsection,
    format=\color{DayanNavy}\bfseries\fontsize{13pt}{15.6pt}\selectfont,
    beforeskip=8pt,
    afterskip=8pt
  },
  paragraph={
    name={},
    number=\theparagraph,
    runin=false,
    format=\color{DayanNavy}\bfseries\fontsize{11.5pt}{13.8pt}\selectfont,
    beforeskip=8pt,
    afterskip=8pt
  }
}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyfoot[C]{\color{DayanMuted}\fontsize{9pt}{11pt}\selectfont\thepage}
\fancypagestyle{plain}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
  \fancyfoot[C]{\color{DayanMuted}\fontsize{9pt}{11pt}\selectfont\thepage}
}

\hypersetup{
  unicode=true,
  colorlinks=true,
  linkcolor=DayanLink,
  urlcolor=DayanLink,
  citecolor=DayanLink,
  bookmarksnumbered=true,
  pdfdisplaydoctitle=true
}

\DeclareCaptionFont{dayancaption}{\color{DayanNavy}\bfseries\fontsize{9.2pt}{12pt}\selectfont}
\captionsetup{
  font=dayancaption,
  labelfont=dayancaption,
  justification=centering,
  singlelinecheck=false,
  labelsep=space,
  skip=5pt
}
\captionsetup[table]{position=top}
\captionsetup[figure]{position=top}
\@addtoreset{figure}{section}
\@addtoreset{table}{section}
\renewcommand{\thefigure}{\thesection.\arabic{figure}}
\renewcommand{\thetable}{\thesection.\arabic{table}}

\DefTblrTemplate{caption-sep}{dayan}{\hspace{0.5em}}
\SetTblrTemplate{caption-sep}{dayan}
\DefTblrTemplate{conthead-text}{dayan}{（续）}
\SetTblrTemplate{conthead-text}{dayan}
\DefTblrTemplate{contfoot-text}{dayan}{续下页}
\SetTblrTemplate{contfoot-text}{dayan}
\SetTblrStyle{caption}{
  fg=DayanNavy,
  font={\bfseries\fontsize{9.2pt}{12pt}\selectfont}
}
\SetTblrStyle{capcont}{
  fg=DayanNavy,
  font={\bfseries\fontsize{9.2pt}{12pt}\selectfont}
}
\SetTblrStyle{contfoot}{
  fg=DayanMuted,
  font={\fontsize{8.5pt}{11pt}\selectfont}
}

\DeclareRobustCommand{\code}[1]{%
  \begingroup
  \color{DayanCode}\fontsize{9pt}{11pt}\selectfont\ttfamily\nolinkurl{#1}%
  \endgroup
}

\NewDocumentEnvironment{dayantable}{m m m}{%
  \begin{longtblr}[
    caption={#1},
    label={#2}
  ]{
    width=\linewidth,
    colspec={#3},
    rowhead=1,
    row{1}={font=\bfseries,fg=DayanNavy,bg=DayanTableHead},
    row{even}={bg=DayanTableStripe},
    cells={font=\fontsize{9.1pt}{13pt}\selectfont,valign=m},
    colsep=6pt,
    rowsep=5pt,
    hlines={0.35pt,DayanTableBorder},
    vlines={0.25pt,DayanTableBorder}
  }
}{%
  \end{longtblr}
}

\newtcolorbox{dayannote}{
  enhanced,
  breakable,
  frame hidden,
  colback=DayanTableStripe,
  borderline west={3pt}{0pt}{DayanRule},
  boxsep=0pt,
  left=9pt,
  right=9pt,
  top=7pt,
  bottom=7pt,
  before skip=8pt,
  after skip=8pt,
  fonttitle=\bfseries\color{DayanNavy},
  coltitle=DayanNavy,
  title=说明
}

\DefineVerbatimEnvironment{dayancode}{Verbatim}{
  breaklines=true,
  breakanywhere=true,
  fontsize=\small,
  frame=single,
  framesep=5pt,
  rulecolor=\color{DayanTableBorder},
  formatcom=\color{DayanCode}
}

\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}
\newcommand{\version}[1]{\gdef\@version{#1}}
\gdef\@subtitle{}
\gdef\@version{unknown}

\renewcommand{\maketitle}{%
  \begin{titlepage}
    \thispagestyle{empty}
    \centering
    \vspace*{0.30\textheight}
    {\color{DayanNavy}\bfseries\fontsize{30pt}{34.5pt}\selectfont\@title\par}
    \vspace{12pt}
    {\color{DayanSecondary}\fontsize{15pt}{19pt}\selectfont\@subtitle\par}
    \vfill
    {\color{DayanSecondary}\fontsize{10.5pt}{16.2pt}\selectfont\@author\par}
    \vspace{6pt}
    {\color{DayanSecondary}\fontsize{10.5pt}{16.2pt}\selectfont
      版本 \@version\quad\@date\par}
  \end{titlepage}
}

\renewcommand{\contentsname}{目录}
