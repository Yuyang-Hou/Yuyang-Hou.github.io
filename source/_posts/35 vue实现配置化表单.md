---
title: vue实现配置化表单
date: 2022-09-07 19:11:58
---

> 记录前端配置话的方案和优化点

### 1、通过协议进行渲染

```html
<template>
  <el-form-item :label="props.label">
    <el-input 
      v-if="props.type === 'el-input' && ...业务联动逻辑" 
      :disabled="props.disabled"
      v-model="props.value"
      ...
    />
    <el-select 
      v-if="props.type === 'el-select' && ...业务联动逻辑" 
      :disabled="props.disabled"
      multiple="props.multiple"
      v-model="props.value"
      ...
    >...</el-select>
  </el-form-item>
</template>
```

> template 中有很多冗余的代码。如果我们需要给组件传入 props 比如例子中的 disabled 、 multiple ；控制 v-if 等等。。

### 2、使用render函数进行代码优化

> render函数组件写法

```html
<script>
export default {
  name: "FormItemDemo",
  render(createElement) {
    return createElement('el-form-item', {
      props: {
        label: '活动名称'
      }
    }, [
        createElement('el-input') // input组件
    ])
  }
}
</script>
```

> render后组件

```html
<script>
export default {
  name: "FormItemDemo",
  props: {
    itemConfig: Object // 接收配置，外部传入
  },
  render(createElement) {
    return createElement('el-form-item', {
      props: {
        label: this.itemConfig.label // 表单项的label
      }
    }, [
        // 表单组件
        createElement(this.itemConfig.type, {
          props: {
            value: this.itemConfig.value // 这里是自己实现一个 v-model
          },
          on: {
            change: (nVal) => { // 这里是自己实现一个 v-model
              this.itemConfig.value = nVal
            }
          }
        }, this.itemConfig.optionData && this.itemConfig.optionData.map(option => {
          // 这里只是本demo 处理 el-select 的 option 数据，实际大家根据具体业务来实现即可
          return createElement('el-option', { props: { label: option.label, value: option.value } })
        }))
    ])
  }
}
</script>
```

> 配置数据

```js
export default [
  {
    type: 'el-input',
    label: '活动名称',
    formKey: 'name',
    value: '', // 默认值为空字符串
    options: {
      vIf: [
        // 表示：当 form.area === 'area1'，才显示
        { relationKey: 'area', value: 'area1' }
      ]
    }
  },
  {
    type: 'el-select',
    label: '活动区域',
    formKey: 'area',
    value: 'area1',
    options: {
      multiple: true
    },
    optionData: [ // 这里模拟去后端拉回数据
      { label: '区域1', value: 'area1' },
      { label: '区域2', value: 'area2' }
    ]
  }
]
```

> 具体使用

```js
<template>
  <div>
    <el-form label-width="100px">
      <FormItemDemo v-for="item in config" :item-config="item" />
    </el-form>
  </div>
</template>

<script>
import FormItemDemo from "./components/FormItemDemo.vue";
import config from "./config";

export default {
  name: 'App',
  components: { FormItemDemo },
  data () {
    return {
      config
    }
  }
}
</script>
```



### 3、配置静态化优化

> 以上方案会滥用vue的响应式，实际上只有input的value需要做响应式处理，这里可以参考vue2源码的做法，将非value属性设置为不可枚举

```js
// 优化函数
function optimize (array) {
  return array.reduce((acc, cur) => {
    for (const key of Object.keys(cur)) {
      if (key === 'value') continue
      // 将不是 value 的属性都进行非响应式优化
      Object.defineProperty(cur, [key], { enumerable: false })
    }
    acc.push(cur)
    return acc
  }, [])
}
```

