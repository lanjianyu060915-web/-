# FastAPI 后端基础框架

这是人脸识别系统课程项目的第二阶段后端框架。在第一阶段的人员与识别记录基础 API、SQLite 初始化逻辑之上，当前增加了人脸照片注册接口，但仍未接入 InsightFace 或其他人脸识别模型。

## 环境要求

- Python 3.10+
- pip

## 安装依赖

建议在 `backend` 目录中创建虚拟环境：

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 启动后端

在 `backend` 目录中运行：

```bash
uvicorn main:app --reload
```

启动后访问：

- API 健康检查：http://127.0.0.1:8000/api/health
- Swagger 文档：http://127.0.0.1:8000/docs

## 数据库

SQLite 数据库会在应用启动时自动创建，路径为：

```text
backend/data/face_system.db
```

## 已实现接口

- `GET /api/health`：健康检查
- `GET /api/persons`：查询人员列表
- `POST /api/persons`：新增人员
- `DELETE /api/persons/{person_id}`：删除人员
- `GET /api/records`：查询识别记录
- `POST /api/persons/{person_id}/faces`：为指定人员注册多张人脸照片，并保存占位人脸特征

## 注册人脸照片接口测试

当前阶段的人脸特征提取仍是可替换的占位实现，不会接入 InsightFace，也不会下载人脸识别模型。接口支持使用 `multipart/form-data` 一次上传多张图片，文件字段名必须为 `files`，路径为 `POST /api/persons/{person_id}/faces`。

先创建一个人员：

```bash
curl -X POST http://127.0.0.1:8000/api/persons \
  -H "Content-Type: application/json" \
  -d '{"name":"张三","person_code":"P001","department":"测试部门"}'
```

假设返回的人员 `id` 为 `1`，再注册人脸照片：

```bash
curl -X POST http://127.0.0.1:8000/api/persons/1/faces \
  -F "files=@/path/to/face-1.jpg" \
  -F "files=@/path/to/face-2.jpg"
```

接口会返回本次保存的人脸特征记录列表，其中 `embedding` 字段目前是模拟特征向量。

## 暂未实现

- 真实人脸特征提取
- InsightFace 模型接入
- 前端接口联调
