# 推送到 GitHub（chelsea11100）

本地已 `git init`、已提交，远程为：

**https://github.com/chelsea11100/centrifuge-tube-rack-vision**

## 1. 在 GitHub 网页建空仓库（只需做一次）

1. 打开 https://github.com/new  
2. **Repository name**：`centrifuge-tube-rack-vision`  
3. 选 **Public**（CV 需要公开链接）  
4. **不要**勾选 “Add a README” / .gitignore（本地已有）  
5. 点 **Create repository**

## 2. 在本机终端推送

```bash
cd ~/Downloads/离心管
git push -u origin main
```

首次会要求登录 GitHub（浏览器或 Personal Access Token）。

### 若提示没有权限，用 Token

1. GitHub → Settings → Developer settings → Personal access tokens  
2. 生成 token（勾选 `repo`）  
3. 推送时密码处粘贴 token：

```bash
git push -u origin main
# Username: chelsea11100
# Password: <粘贴 token>
```

## 3. 推送成功后

仓库地址（可写进 CV）：

**https://github.com/chelsea11100/centrifuge-tube-rack-vision**

主页：https://github.com/chelsea11100
