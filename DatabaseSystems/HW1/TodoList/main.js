import Vue from 'vue'
import App from './App'

Vue.config.productionTip = false

App.mpType = 'app'

// window.Vue = Vue

const app = new Vue({
    ...App
})
app.$mount()
