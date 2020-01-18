<template>
<view>
	<view class="title">&emsp;&emsp;ToDoList</view>
	<view class="search_box">
		<span class="custom-font">&#xe8ef;</span>
		<!-- <view class="search_title">查找</view> -->
		<input class="input" type="text" placeholder="查找一个日程"
			@confirm="find" v-model="text_find" />
	</view>
	<view class="list-container">
	<view class="item" v-if="tmp != -1">
		<checkbox :checked="arr[tmp].done" @click="change(tmp)" style="transform: scale(0.7)"/>
		<view :class="[arr[tmp].done ? 'done' : '']" v-if="!arr[tmp].done">
			<input v-model="arr[tmp].value" @confirm="edit(tmp)"/>
		</view>
		<view v-else class="done">{{ arr[tmp].value }}</view>
		<view class="icon-pack">
		<view class="icon-right" @click="star(tmp)">
			<view v-show="arr[tmp].star">&#xe86a;</view>
			<view v-show="!arr[tmp].star">&#xe7df;</view>
		</view>
		<view class="icon-right" @click="del(tmp)">&#xe7c3;</view>
		</view>
	</view>
	</view>
	<view class="search_box">
		<!-- <view class="search_title">已计划日程</view> -->
		<!-- <uni-icon type="plus" size="20"></uni-icon> -->
		<span class="custom-font">&#xe781;</span>
		<input class="input" type="text" placeholder="添加一个日程"
			@confirm="schedule" v-model="text" />
	</view>
	<view class="list-container">
		<view class="item" v-for="(item, index) in arr" :key="index" >
			<checkbox :checked="item.done" @click="change(index)" style="transform: scale(0.7)"/>
			<view :class="[item.done ? 'done' : '']" v-if="!item.done">
				<!-- {{ index + 1 }}.&ensp;{{ item.value }} -->
				<input v-model="arr[index].value" @confirm="edit(index)"/>
			</view>
			<view v-else class="done">{{ item.value }}</view>
			<span class="middle-space"></span>
			<view class="icon-pack">
			<view class="icon-right" @click="star(index)">
				<view v-show="item.star">&#xe86a;</view>
				<view v-show="!item.star">&#xe7df;</view>
			</view>
			<view class="icon-right" @click="del(index)">&#xe7c3;</view>
			</view>
		</view>
	</view>
	<view class="search_box">
		<span class="custom-font">&#xe86a;</span>星标事项
	</view>
	<view class="list-container">
		<view class="item" v-for="(item, index) in starArr" :key="index" >
			<checkbox :checked="item.done" @click="change(index)" style="transform: scale(0.7)"/>
			<view :class="[item.done ? 'done' : '']" v-if="!item.done">
				<!-- {{ index + 1 }}.&ensp;{{ item.value }} -->
				<input v-model="arr[index].value" @confirm="edit(index)"/>
			</view>
			<view v-else class="done">{{ item.value }}</view>
			<view class="icon-pack">
			<view class="icon-right" @click="star(index)">
				<view v-show="item.star">&#xe86a;</view>
				<view v-show="!item.star">&#xe7df;</view>
			</view>
			<view class="icon-right" @click="del(index)">&#xe7c3;</view>
			</view>
		</view>
	</view>
</view>
</template>

<!-- 1.	路由：/pages/index；
2.	可以添加、删除、编辑、保存及读取待办事项；
3.	事件最少包含两种状态：待完成、已完成，且状态可修改；
加分项：
1.	代码整洁规范；
2.	样式美观；
3.	附加其他功能及操作，如查找、状态筛选等等； -->

<script>
import uniIcon from "../../components/uni-icon.vue"
export default {
	data() {
		return {
			tmp: -1,
			arr: [],
			name: '',
			text: '',
			text_find: ''
		}
	},
	components: {
		uniIcon
	},
	onLoad() {
		uni.getStorage({
			key: 'storage_key',
			success: (res) => {
					console.log(res.data);
					this.arr = res.data;
			} // binding
		});
	},
	methods: {
		/**
		* 删除
		* @param {Object} index
		*/
		del(index) {
			this.arr.splice(index, 1);
			uni.setStorage({ // 注意要存储进本地
				key: 'storage_key',
				data: this.arr,
				success: function(){
					console.log('success');
				}
			});
			this.tmp = -1;
		},
		/**
		* 改变状态
		* @param {Number} index
		*/
		change(index) {
			this.arr[index].done = !this.arr[index].done;
			uni.setStorage({ // store the items
				key: 'storage_key',
				data: this.arr,
				success: function(){
					console.log('success');
				}
			});
		},
		/**
		 * 编辑日程，注意只有未被删除的日程才可以被修改
		 * @param {Object} index
		 */
		edit(index) {
			uni.setStorage({ // store the items
				key: 'storage_key',
				data: this.arr,
				success: function(){
					console.log('success');
				}
			});
		},
		/**
		* 添加日程
		* @param {Object} e
		*/
		schedule(e) {
			let value = e.detail.value;
			if (value == '') {
				console.log('None!');
				return;
			}
			this.arr.push({
				value: value,
				done: false,
				star: false
			});
			console.log(this.arr);
			this.text = '';
			uni.setStorage({
				key: 'storage_key',
				data: this.arr,
				success: function(){
					console.log('success');
				}
			})
			this.tmp = -1; // 确保查询面板清空
		},
		/**
		 * 查找某一日程并显示，注意共享数据关系
		 * @param {Object} e 
		 */
		find(e) {
			this.tmp = -1;
			for (var i = 0; i < this.arr.length; i++) {
				if (this.arr[i].value == e.detail.value) {
					this.text = '';
					this.text_find = '';
					this.tmp = i;
					break;
				}
			}
		},
		star(index) {
			this.arr[index].star = !this.arr[index].star;
			uni.setStorage({
				key: 'storage_key',
				data: this.arr,
				success: function(){
					console.log('success');
				}
			})
		}
	},
	computed: {
		starArr: function () {
			return this.arr.filter(function (item) {
				return item.star
			})
		}
	}
}
</script>

<style>
@import '../../common/iconfont.css';
/* @import '../../common/uni.css'; */
page {
	  background-color: #315481;
	  background-image: linear-gradient(to bottom, #315481, #918e82 100%);
	  background-repeat: no-repeat;
	  background-attachment: fixed;
}
.title {
	height: 80rpx;
	text-align: left;
	color: #999999;
	font-size: 50rpx;
	background: rgba(47,47,47,0.98);
	display: block;
	/* up right down left */
	/* padding: 10rpx 0rpx 5rpx 20rpx; */
	padding: 10rpx 20rpx;
}
.search_box {
	display: flex;
	flex-wrap: wrap;
	align-items: center;
	background: linear-gradient(180deg,#d0edf5,#e1e5f0);
	width: 95%;
	height: 80rpx;
	margin: 20rpx auto;
	/* margin-bottom: 10rpx; */
}
/* .search_title {
	font-size: 45rpx;
	text-align: left;
} */
.custom-font {
    font-family: iconfont;
	/* font-size: 15rpx; */
    margin: 10rpx;
}
.input {
	cursor: text;
	border-bottom: 0.5px solid #000;
}

.list-container {
	width: 95%;
	/* height: 850rpx; */
	margin: 20rpx auto;
	background-color: #EEEEEE;
}
.item {
	display: flex;
	display: -webkit-flex;
	flex-wrap: wrap;
	align-items: center;
	font-size: 35rpx;
	height: 85rpx;
	margin: 0 20rpx;
	background-color: #EEEEEE;
}
.done {
	text-decoration: line-through;
	opacity: .5;
	color: #999;
}
.icon-pack {
	width: 100rpx;
	-webkit-flex: 1;
	flex: 1;
	justify-content: space-between;
}
.icon-right {
	font-family: iconfont;
	font-size: 45rpx;
	color: #000000;
	margin: 5rpx;
	text-align: right;
	float: right;
}
</style>
