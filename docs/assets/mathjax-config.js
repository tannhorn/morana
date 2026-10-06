window.MathJax = {
  options: {
    enableSpeech: false,
    enableBraille: false,
  },
  loader: {
    load: ["[tex]/boldsymbol"],
  },
  menuOptions: {
    settings: {
      speech: false,
      braille: false,
    },
  },
  output: {
    font: "mathjax-newcm",
    // Resolve from this script so nested documentation pages use the same fonts.
    fontPath: new URL("mathjax/font", document.currentScript.src).href,
  },
  tex: {
    inlineMath: {
      "[+]": [["$", "$"]],
    },
    packages: {
      "[+]": ["boldsymbol"],
    },
  },
};
