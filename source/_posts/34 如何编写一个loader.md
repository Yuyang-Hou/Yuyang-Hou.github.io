---
title: 如何编写一个简单的loader
date: 2022-09-06 19:11:58
---

> 一直都知道loader是文件的翻译器，那么如何实现一个简单的loader并跑通呢？
>
> 尝试实现一个loader翻译md文本
>
> 收获：最后返回的东西一定要是js

```
// markdown-loader.js
// 对外暴露一个函数
const {marked} = require('marked')
module.exports = (source) => {
    // 输入加载的资源
    const html = marked(source)
    // 不能返回html，一定要是js
    // 处理资源
    const code = `moudle.exports=${JSON.stringify(html)}`

    // htmlloader就是把html处理成js的loader

    // 处理完的一定是要是一个js代码，不然在输出中的无法执行
    return code
}
```

```
// webpack.config.js
const path  = require('path')
module.exports = {
    mode: 'none',
    entry: "./src/main.js",
    output: {
        filename: 'bundle.js'
    },	

    module: {
        rules:[
            {
                test: /\.md$/,
                use: './markdown-loader'
            }
        ]
    }
}
```

