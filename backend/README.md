# FastAPI 后端基础框架

这是人脸识别系统课程项目的第四阶段后端框架。在第一阶段的人员与识别记录基础 API、第二阶段的人脸照片注册接口、第三阶段占位 embedding 识别接口之上，当前阶段接入了 `insightface` + `onnxruntime`，用于真实人脸检测和 embedding 提取。

## 环境要求

- Python 3.10+
- pip
- 首次运行模型时需要能够访问 InsightFace 模型下载地址

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

本阶段新增的人脸识别依赖包括：

- `insightface`：人脸检测和人脸 embedding 提取。
- `onnxruntime`：运行 InsightFace ONNX 模型。
- `opencv-python`、`numpy`：图片解码和矩阵处理。

如果只想手动安装核心依赖，也可以在虚拟环境中运行：

```bash
pip install insightface onnxruntime opencv-python numpy
```

## InsightFace 模型下载说明

后端第一次调用注册或识别接口时，会懒加载 InsightFace `buffalo_l` 模型。模型文件可能在首次运行时自动下载，因此首次请求会比后续请求慢。

如果模型下载失败或模型初始化失败，服务不会崩溃；接口会返回清晰的 `503` 错误，提示检查依赖安装和模型下载网络。可以按下面顺序排查：

1. 确认已在 `backend` 虚拟环境中执行 `pip install -r requirements.txt`。
2. 确认运行后端的机器可以访问 InsightFace 模型下载地址。
3. 重新启动后端，再次调用注册或识别接口触发模型加载。
4. 如果生产环境不能联网，请提前在可联网环境下载模型，并把模型缓存目录复制到部署环境的 InsightFace 缓存路径。

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

人脸 embedding 仍保存到 `face_embeddings.embedding_json` 字段，但内容已经从占位向量改为真实 InsightFace embedding。

## 已实现接口

- `GET /api/health`：健康检查
- `GET /api/persons`：查询人员列表
- `POST /api/persons`：新增人员
- `DELETE /api/persons/{person_id}`：删除人员
- `GET /api/records`：查询最近识别记录
- `POST /api/persons/{person_id}/faces`：为指定人员注册多张人脸照片，检测人脸并保存真实 embedding
- `POST /api/recognize`：上传一张图片，检测每张人脸并与已注册 embedding 比对，返回最佳匹配结果并写入识别记录

## 注册人脸照片接口测试

接口支持使用 `multipart/form-data` 一次上传多张图片，文件字段名必须为 `files`，路径为 `POST /api/persons/{person_id}/faces`。

注册时会检测图片中的人脸：

- 未检测到人脸：返回 `422` 和清晰提示。
- 检测到多张人脸：自动选择面积最大的人脸用于注册。
- 模型或依赖不可用：返回 `503` 和初始化失败说明，不会导致服务崩溃。

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

接口会返回本次保存的人脸特征记录列表，其中 `embedding` 字段是真实模型提取的向量。

## 识别人脸接口测试

识别接口使用 `multipart/form-data` 上传单张图片，文件字段名必须为 `file`，路径为 `POST /api/recognize`。

```bash
curl -X POST http://127.0.0.1:8000/api/recognize \
  -F "file=@/path/to/recognize.jpg"
```

识别时会检测上传图片中的每张人脸，分别提取 embedding，并与 `face_embeddings` 表中保存的 embedding 逐个计算余弦相似度，最终返回全局最佳匹配：

- `name`：识别到的人员姓名；未达到阈值时为 `未知`。
- `person_code`：识别到的人员编号；未达到阈值时为 `null`。
- `similarity`：与库中最佳匹配人脸特征的余弦相似度；当前为空库时为 `null`。
- `status`：识别状态，`recognized` 表示相似度达到阈值，`unknown` 表示未达到阈值或暂无已注册人脸。

识别完成后可以查询最近识别记录：

```bash
curl http://127.0.0.1:8000/api/records
```

## 暂未实现

- 前端接口联调
- 生产环境模型缓存管理
