// .cjs 확장자: 프로젝트가 "type": "module" 이지만 PostCSS가 CommonJS 설정을 기대.
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
