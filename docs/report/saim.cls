% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sAIm article class
% Cleaned template version derived from the uploaded project
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{saim}[2026/03/22 Cleaned sAIm manuscript template]
\LoadClassWithOptions{article}

% Layout
\RequirePackage[top=2.25cm,bottom=2.5cm,left=2.5cm,right=2.5cm,columnsep=0.65cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[english]{babel}
\RequirePackage{etoolbox}
\DisableLigatures[f]{family=sf*}

% Graphics
\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables and math helpers
\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colors
\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
\definecolor{saimindigo}{HTML}{32127A}
\definecolor{saimfg}{HTML}{32127A}
\definecolor{saimbg}{HTML}{D6CFE4}

% References
\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=saimindigo,
  citecolor=saimindigo,
  urlcolor=saimindigo
}
\RequirePackage[round,authoryear]{natbib}
\def\bibfont{\small}

% Section and caption formatting
\RequirePackage{titlesec}
\titleformat*{\paragraph}{\itshape}
\titleformat*{\section}{\Large\sffamily\bfseries}
\titleformat*{\subsection}{\large\sffamily\bfseries}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% Fonts
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\renewcommand*\ttdefault{cmvtt}

% Metadata lists
\makeatletter
\renewcommand\addtolist[5][]{%
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}
\makeatother

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily\bfseries #2$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{, }}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\small $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily\bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

\newcommand\abstractlist{}
\renewcommand{\abstract}[1]{\gdef\abstractlist{{\color{saimfg} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}
\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}
\renewcommand{\title}[1]{\gdef\titlelist{{\huge\sffamily\bfseries #1}}}
\newcommand{\titlelist}{}

\newcommand{\beginappendix}{\appendix\par\bigskip{\huge\sffamily Appendix\par\medskip}}

% Title box
\newcommand{\mymaketitle}{%
  \tcbset{enhanced,frame hidden,left=0.5cm,right=0.5cm,top=0.5cm,bottom=0.5cm,arc=10pt,colback=saimbg,before skip=0pt,grow to left by=1.5pt,grow to right by=1.5pt}
  \tcbset{overlay={\node[anchor=south east,at=(frame.south east),xshift=-0.5cm,yshift=0.2cm]{\includegraphics[width=1.8cm]{assets/saim_logo.png}};}}
  \begin{tcolorbox}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.5cm}
    {%
      \setlength{\parskip}{0pt}
      \raggedright
      \nohyphens
      {\setstretch{1.25}\titlelist\par}
      \vskip 0.2cm
      \authorlist\par
      \vskip 0.15cm
      \affiliationlist\par
      \contributionlist\par
    }
    \ifstrempty{\abstractlist}{}{\vskip 0.35cm\abstractlist\par}
    \vskip 0.35cm
    {\setlength{\parskip}{0cm}\ifdefempty{\metadatalist}{}{\metadatalist\par}}
  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\if@twocolumn
\renewcommand{\maketitle}{%
\twocolumn[%
  \mymaketitle
  \vskip 0.38cm
]}
\else
\renewcommand{\maketitle}{\mymaketitle}
\fi
