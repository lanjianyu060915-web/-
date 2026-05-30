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

## 暂未实现

- 人脸图片上传
- 人脸特征提取
- InsightFace 模型接入
- 前端接口联调
