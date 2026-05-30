# FastAPI 后端基础框架

这是人脸识别系统课程项目的第一阶段后端框架。当前只包含人员与识别记录的基础 API、SQLite 初始化逻辑，暂未接入 InsightFace 或其他人脸识别模型。

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
- `POST /api/persons/{person_id}/faces`：为指定人员注册一张或多张人脸照片（当前使用模拟特征向量）

## 暂未实现

- 独立图片识别上传
- 真实人脸特征提取
- InsightFace 模型接入
- 前端接口联调

## 注册人脸照片

当前阶段新增了人脸照片注册接口，但仍然使用可替换的模拟特征向量，尚未接入 InsightFace 或其他真实人脸识别模型。

接口：

```text
POST /api/persons/{person_id}/faces
```

请求类型为 `multipart/form-data`，字段名为 `files`，支持一次上传多张图片：

```bash
curl -X POST "http://127.0.0.1:8000/api/persons/1/faces" \
  -F "files=@/path/to/face-1.jpg" \
  -F "files=@/path/to/face-2.png"
```

返回示例：

```json
{
  "person_id": 1,
  "registered_count": 2,
  "faces": [
    {
      "id": 1,
      "person_id": 1,
      "file_name": "face-1.jpg",
      "content_type": "image/jpeg",
      "embedding_dimension": 128,
      "created_at": "2026-05-30 12:00:00"
    }
  ]
}
```

后续接入真实模型时，可优先替换 `face_service.py` 中的 `extract_embedding(image)` 函数。
