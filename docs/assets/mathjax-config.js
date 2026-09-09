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
    fontPath: "assets/mathjax/font",
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
