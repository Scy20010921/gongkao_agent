<template>
  <div id="app">
    <div class="chat-container">
      <!-- 头部 -->
      <div class="header">
        <h1>📚 公考智能助手</h1>
        <span class="status" :class="{ online: isOnline }">
          {{ isOnline ? '● 在线' : '● 离线' }}
        </span>
      </div>

      <!-- 消息列表 -->
      <div class="messages" ref="messagesContainer">
        <div v-for="(msg, index) in messages" :key="index" 
             :class="['message', msg.role]">
          <div class="avatar">
            {{ msg.role === 'user' ? '👤' : '🤖' }}
          </div>
          <div class="content" v-html="renderMarkdown(msg.content)"></div>
        </div>
        <div v-if="loading" class="message assistant">
          <div class="avatar">🤖</div>
          <div class="content typing">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="input-area">
        <input 
          v-model="inputText" 
          @keydown.enter="sendMessage"
          placeholder="请输入你的问题..."
          :disabled="loading"
        />
        <button @click="sendMessage" :disabled="loading || !inputText.trim()">
          {{ loading ? '思考中...' : '发送' }}
        </button>
      </div>

      <!-- 快捷指令 -->
      <div class="quick-actions">
        <button @click="quickSend('生成行测每日一练')">📝 每日一练</button>
        <button @click="quickSend('帮我规划学习计划')">📋 学习规划</button>
        <button @click="quickSend('推荐岗位')">💼 选岗推荐</button>
        <button @click="quickSend('批改这篇申论：')">📖 申论批改</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue'
import axios from 'axios'
import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'

// ===== 状态 =====
const inputText = ref('')
const messages = ref([])
const loading = ref(false)
const isOnline = ref(false)
const messagesContainer = ref(null)

// ===== 配置 =====
const API_BASE = '/api'
// 如果你直接访问后端，可以用 http://127.0.0.1:8000
// const API_BASE = 'http://127.0.0.1:8000'

// ===== 方法 =====
const renderMarkdown = (text) => {
  if (!text) return ''
  return marked.parse(text, {
    highlight: (code, lang) => {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value
      }
      return hljs.highlightAuto(code).value
    }
  })
}

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const sendMessage = async () => {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  // 添加用户消息
  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  loading.value = true
  scrollToBottom()

  try {
    const response = await axios.post(`${API_BASE}/chat`, {
      question: text
    }, {
      timeout: 180000  // 3分钟超时
    })

    const data = response.data
    const answer = data.answer || '抱歉，未获取到回答。'
    const intent = data.intent || ''

    // 构建带意图标签的回答
    let displayAnswer = answer
    if (intent) {
      const intentMap = {
        'qa': '💬 答疑',
        'exam': '📝 出题',
        'plan': '📋 规划',
        'grading': '📖 批改',
        'job': '💼 选岗',
        'ops': '📢 运营'
      }
      displayAnswer = `**${intentMap[intent] || intent}**\n\n${answer}`
    }

    messages.value.push({ role: 'assistant', content: displayAnswer })
  } catch (error) {
    console.error('请求失败:', error)
    messages.value.push({ 
      role: 'assistant', 
      content: '❌ 请求失败，请检查后端服务是否运行。\n\n错误信息：' + error.message 
    })
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

const quickSend = (text) => {
  if (text.includes('批改这篇申论：')) {
    inputText.value = text
  } else {
    inputText.value = text
    sendMessage()
  }
}

// ===== 检查后端状态 =====
const checkHealth = async () => {
  try {
    await axios.get(`${API_BASE}/health`, { timeout: 3000 })
    isOnline.value = true
  } catch {
    isOnline.value = false
  }
}

// ===== 生命周期 =====
onMounted(() => {
  checkHealth()
  // 每30秒检查一次状态
  setInterval(checkHealth, 30000)
})
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #f0f2f5;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}

.chat-container {
  width: 100%;
  max-width: 900px;
  height: 95vh;
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.header {
  padding: 16px 24px;
  background: linear-gradient(135deg, #1a1a2e, #16213e);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.header h1 {
  font-size: 20px;
  font-weight: 600;
}

.status {
  font-size: 12px;
  padding: 4px 12px;
  border-radius: 20px;
  background: #ff4757;
  color: white;
  transition: all 0.3s;
}

.status.online {
  background: #2ed573;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #fafafa;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  max-width: 85%;
}

.message.user {
  margin-left: auto;
  flex-direction: row-reverse;
}

.message .avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.message.user .avatar {
  background: #4a6cf7;
  color: white;
}

.message.assistant .avatar {
  background: #f0f0f0;
}

.message .content {
  padding: 12px 16px;
  border-radius: 12px;
  background: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  line-height: 1.6;
  font-size: 14px;
  overflow-wrap: break-word;
  word-break: break-word;
}

.message.user .content {
  background: #4a6cf7;
  color: white;
  border-bottom-right-radius: 4px;
}

.message.assistant .content {
  background: white;
  border-bottom-left-radius: 4px;
}

.message .content :deep(pre) {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 8px 0;
}

.message .content :deep(code) {
  font-family: 'Consolas', monospace;
  font-size: 13px;
}

.typing {
  display: flex;
  gap: 4px;
  padding: 8px 0;
}

.typing span {
  width: 8px;
  height: 8px;
  background: #999;
  border-radius: 50%;
  animation: typing 1.4s infinite both;
}

.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}

.input-area {
  display: flex;
  gap: 12px;
  padding: 16px 20px;
  border-top: 1px solid #eee;
  background: white;
  flex-shrink: 0;
}

.input-area input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ddd;
  border-radius: 24px;
  outline: none;
  font-size: 14px;
  transition: border-color 0.3s;
}

.input-area input:focus {
  border-color: #4a6cf7;
}

.input-area button {
  padding: 12px 28px;
  background: #4a6cf7;
  color: white;
  border: none;
  border-radius: 24px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s;
  flex-shrink: 0;
}

.input-area button:hover:not(:disabled) {
  background: #3a56d4;
}

.input-area button:disabled {
  background: #aaa;
  cursor: not-allowed;
}

.quick-actions {
  display: flex;
  gap: 8px;
  padding: 8px 20px 16px;
  background: white;
  border-top: 1px solid #f0f0f0;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.quick-actions button {
  padding: 6px 14px;
  background: #f0f4ff;
  color: #4a6cf7;
  border: 1px solid #e0e8ff;
  border-radius: 20px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.quick-actions button:hover {
  background: #e0e8ff;
  transform: translateY(-1px);
}
</style>