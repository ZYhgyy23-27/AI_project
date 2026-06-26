# 微信小程序嵌入「IMU 姿态可视化」交付说明

本文档供 **小程序开发者** 与 **运维/后端** 分工使用：前者按第 4 节建页面；后者按第 2～3 节部署 HTTPS 并在微信公众平台配置域名。

---

## 1. 接口与约定（给双方对齐）

| 项目 | 说明 |
|------|------|
| H5 页面 URL | `https://<你的域名>/attitude` |
| WebSocket | 与页面同源：`wss://<你的域名>/ws/attitude`（页面在 HTTPS 下会自动使用 `wss`） |
| 下行数据 | JSON 文本帧，含四元数字段 `w, x, y, z`（可选 `seq`、`t_us`） |
| Three.js | 同源脚本 `https://<你的域名>/attitude/vendor/three.min.js`（无需再配置 CDN 域名） |

小程序侧采用 **`web-view` 全屏打开上述 H5**，不在小程序内重写 3D。

---

## 2. 服务器部署要点（运维 / 后端）

### 2.1 必须 HTTPS + WSS

微信要求 `web-view` 的地址为 **https**，且页面内 WebSocket 在真机上应为 **wss**。请在公网入口使用 **Nginx / Caddy** 等设备在 **443** 端口终结 TLS，再反代到本机 Flask（例如 `127.0.0.1:5000`）。

**不要将 Flask 端口直接暴露为纯 HTTP 给公网用户**，应只暴露 443。

### 2.2 Nginx 反代示例（含 WebSocket）

将 `viz.example.com` 换成实际域名；上游端口与线上一致。

```nginx
server {
    listen 443 ssl http2;
    server_name viz.example.com;

    ssl_certificate     /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

部署完成后，用**手机浏览器**访问 `https://viz.example.com/attitude`，确认 3D 与连接状态正常。

### 2.3 微信「业务域名」校验文件

1. 登录 [微信公众平台](https://mp.weixin.qq.com/) → 小程序 → **开发管理** → **开发设置** → **业务域名**，按提示下载 `MP_verify_xxxxxxxx.txt`。
2. 将该文件放到服务器可读位置，启动服务前设置环境变量：

```bash
export WECHAT_MP_VERIFY_FILE_PATH=/path/to/MP_verify_xxxxxxxx.txt
```

3. 重启 Flask 应用后，应能通过浏览器访问：

   `https://viz.example.com/MP_verify_xxxxxxxx.txt`

   打开后为**纯文本**，内容与微信下载文件一致。

若校验文件由 Nginx 静态目录单独提供、不经过 Flask，则无需设置该环境变量，但需保证 URL 与微信要求一致。

---

## 3. 微信公众平台配置（小程序管理员）

路径：**开发** → **开发管理** → **开发设置**。

1. **业务域名**  
   添加：`viz.example.com`（不要带 `https://`）。按流程完成校验文件或 DNS 校验。

2. **socket 合法域名**  
   添加：`viz.example.com`。  
   页面在 HTTPS 下会连接 `wss://viz.example.com/ws/attitude`，与此一致。

3. **request 合法域名**（按需）  
   当前 H5 的 Three.js 已改为**同源** `/attitude/vendor/three.min.js`，一般只需业务域名本身已备案在服务端即可；若 H5 后续增加其它域名的 `https` 请求，需在此补充对应域名。

保存后 **等待生效**（通常几分钟级，以微信侧为准）。

---

## 4. 小程序工程实现（给小程序开发者）

### 4.1 前置条件

- 已注册小程序，且 **AppID** 可用。
- 已安装 [微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)。
- **业务域名**、**socket 合法域名** 已按第 3 节配置完成。
- 已知姿态页完整地址，例如：`https://viz.example.com/attitude`。

### 4.2 新建页面（示例：`pages/attitude/attitude`）

在小程序根目录 `app.json` 的 `pages` 数组中**注册该页面**（路径按你项目实际调整）：

```json
{
  "pages": [
    "pages/index/index",
    "pages/attitude/attitude"
  ]
}
```

### 4.3 页面 JSON（可选全屏网页）

`pages/attitude/attitude.json`：

```json
{
  "navigationBarTitleText": "实时姿态",
  "navigationStyle": "custom"
}
```

若希望保留系统导航栏，删除 `navigationStyle` 一行即可。

### 4.4 页面 WXML

`pages/attitude/attitude.wxml`：

```xml
<web-view src="{{url}}"></web-view>
```

### 4.5 页面逻辑

`pages/attitude/attitude.js`：

```javascript
Page({
  data: {
    url: 'https://viz.example.com/attitude'
  }
});
```

将 `viz.example.com` 换成实际域名。**不要**在代码里写 `http://` 或未备案域名。

若需带查询参数（例如将来做简单令牌，需与后端约定），可在 `onLoad` 里拼接，注意不要把敏感密钥写死在前端。

### 4.6 从其它页跳转

```javascript
wx.navigateTo({
  url: '/pages/attitude/attitude'
});
```

### 4.7 开发者工具设置

- 真机预览前：在开发者工具 **详情** → **本地设置**，可勾选「不校验合法域名、web-view、TLS 版本…」**仅用于本机调试**；**正式体验与上架必须以合法域名为准**。
- 首次验证建议在 **真机预览** 下测试 `web-view` 与 `wss`。

---

## 5. 联调检查清单

- [ ] `https://域名/attitude` 在手机浏览器正常，WebSocket 显示已连接。
- [ ] `https://域名/MP_verify_*.txt` 可访问且内容正确（若走 Flask 校验文件）。
- [ ] 微信公众平台业务域名、socket 合法域名已添加并生效。
- [ ] 小程序 `web-view` 的 `src` 与业务域名一致（https）。
- [ ] 真机预览：姿态块随设备数据转动（需 MQTT 有 `esp32/attitude` 数据）。

---

## 6. 常见问题

| 现象 | 处理方向 |
|------|----------|
| web-view 空白 | 查业务域名、证书、是否 https；看开发者工具控制台报错 |
| 页面能开但一直连接中 | 查 socket 合法域名、Nginx 是否转发 `Upgrade`、是否实际使用 wss |
| 个人主体小程序 | 以微信最新规则为准，部分能力对个人主体有限制 |

---

## 7. 代码变更摘要（仓库内，供后端自查）

- `dashboard.html`：Three.js 改为同源 `/attitude/vendor/three.min.js`。
- `attitude_viz/server.py`：注册该静态路径；可选 `WECHAT_MP_VERIFY_FILE_PATH` 提供根路径校验文件。
- `static/vendor/three.min.js`：随仓库提供（Three.js r128），避免依赖外网 CDN。
