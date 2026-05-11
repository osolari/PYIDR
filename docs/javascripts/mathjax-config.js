window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      RR: "{\\mathbb{R}}",
      EE: "{\\mathbb{E}}",
      PP: "{\\mathbb{P}}",
      ind: "{\\mathbf{1}}",
      bm: ["{\\boldsymbol{#1}}", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
