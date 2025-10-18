"""
FastAPI Main Application Module
Provides REST API interface for sports injury risk prediction
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import logging
import os
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib

from ..data.contracts import PlayerDataContract, InjuryPredictionResult
from ..core.trainer import BaseTrainer
from ..core.metrics import MetricsCalculator
from ..data.loader import DataLoader
from ..data.features import FeatureEngineer
from ..data.validate import DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Sports Injury Risk Prediction API",
    description="API for predicting sports injury risk using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should set specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
security = HTTPBearer(auto_error=False)

# Global variables for model storage
loaded_models: Dict[str, Any] = {}
feature_engineer: Optional[FeatureEngineer] = None
data_validator: Optional[DataValidator] = None


# Pydantic model definitions
class PlayerInput(BaseModel):
    """Single player input data"""
    player_id: str = Field(..., description="Player ID")
    age: int = Field(..., ge=16, le=50, description="Age")
    position: str = Field(..., description="Position")
    height: float = Field(..., gt=0, description="Height (cm)")
    weight: float = Field(..., gt=0, description="Weight (kg)")
    games_played: int = Field(..., ge=0, description="Games played")
    minutes_played: int = Field(..., ge=0, description="Total minutes played")
    recent_injury: bool = Field(False, description="Recent injury status")
    injury_history: List[str] = Field(default_factory=list, description="Injury history")
    training_load: Optional[float] = Field(None, description="Training load")
    match_intensity: Optional[float] = Field(None, description="Match intensity")

    @validator('position')
    def validate_position(cls, v):
        valid_positions = ['GK', 'DEF', 'MID', 'FWD']
        if v not in valid_positions:
            raise ValueError(f'Position must be one of {valid_positions}')
        return v


class BatchPlayerInput(BaseModel):
    """Batch player input data"""
    players: List[PlayerInput] = Field(..., description="List of player data")
    model_name: Optional[str] = Field("default", description="Model name to use")


class PredictionResponse(BaseModel):
    """Prediction response"""
    player_id: str
    injury_risk: float = Field(..., ge=0, le=1, description="Injury risk probability")
    risk_level: str = Field(..., description="Risk level")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    contributing_factors: Dict[str, float] = Field(default_factory=dict, description="Risk factor contributions")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any] = Field(default_factory=dict, description="批量预测汇总")
    processing_time: float = Field(..., description="处理时间(秒)")


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    type: str
    version: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    feature_names: List[str]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: datetime
    loaded_models: List[str]
    memory_usage: Dict[str, Any]


# 辅助函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户（简单的认证示例）"""
    if credentials is None:
        return None

    # 这里可以添加真实的JWT验证逻辑
    token = credentials.credentials
    if token == "demo-token":  # 示例token
        return {"user_id": "demo_user", "permissions": ["predict", "batch_predict"]}

    return None


def load_models():
    """加载预训练模型"""
    global loaded_models, feature_engineer, data_validator

    models_dir = Path(os.getenv("MODELS_DIR", "models"))

    if not models_dir.exists():
        logger.warning(f"Models directory {models_dir} not found")
        return

    # 加载默认模型
    default_model_path = models_dir / "default_model.joblib"
    if default_model_path.exists():
        try:
            loaded_models["default"] = joblib.load(default_model_path)
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading default model: {e}")

    # 加载其他模型
    for model_file in models_dir.glob("*.joblib"):
        if model_file.stem != "default_model":
            try:
                model_name = model_file.stem
                loaded_models[model_name] = joblib.load(model_file)
                logger.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")

    # 初始化特征工程器和验证器
    try:
        feature_engineer = FeatureEngineer()
        data_validator = DataValidator()
        logger.info("Feature engineer and data validator initialized")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")


def calculate_risk_level(risk_score: float) -> str:
    """计算风险等级"""
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def generate_recommendations(risk_score: float, contributing_factors: Dict[str, float]) -> List[str]:
    """生成建议措施"""
    recommendations = []

    if risk_score > 0.7:
        recommendations.append("建议减少训练强度")
        recommendations.append("增加恢复时间")
    elif risk_score > 0.5:
        recommendations.append("密切监控身体状况")
        recommendations.append("调整训练计划")

    # 基于贡献因子的具体建议
    top_factors = sorted(contributing_factors.items(), key=lambda x: x[1], reverse=True)[:3]

    for factor, score in top_factors:
        if factor == "training_load" and score > 0.1:
            recommendations.append("考虑降低训练负荷")
        elif factor == "match_intensity" and score > 0.1:
            recommendations.append("控制比赛强度")
        elif factor == "recent_injury" and score > 0.1:
            recommendations.append("加强伤病部位的康复训练")

    return recommendations


# API端点
@app.get("/", response_model=Dict[str, str])
async def root():
    """根端点"""
    return {
        "message": "Sports Injury Risk Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    import psutil

    memory_info = psutil.virtual_memory()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        loaded_models=list(loaded_models.keys()),
        memory_usage={
            "total": memory_info.total,
            "available": memory_info.available,
            "percent": memory_info.percent,
            "used": memory_info.used
        }
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """列出所有可用模型"""
    model_infos = []

    for name, model in loaded_models.items():
        try:
            # 尝试获取模型信息
            model_info = ModelInfo(
                name=name,
                type=type(model).__name__,
                version="1.0",
                created_at=datetime.now(),  # 实际应该从模型元数据获取
                performance_metrics={},  # 实际应该从模型元数据获取
                feature_names=getattr(model, 'feature_names_in_', [])
            )
            model_infos.append(model_info)
        except Exception as e:
            logger.error(f"Error getting info for model {name}: {e}")

    return model_infos


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    player_data: PlayerInput,
    model_name: str = "default",
    user=Depends(get_current_user)
):
    """单个球员伤病风险预测"""

    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = loaded_models[model_name]

    try:
        # 转换输入数据为DataFrame
        player_df = pd.DataFrame([player_data.dict()])

        # 数据验证
        if data_validator:
            validation_result = data_validator.validate_dataframe(player_df)
            if not validation_result['is_valid']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Data validation failed: {validation_result['errors']}"
                )

        # 特征工程
        if feature_engineer:
            player_features = feature_engineer.transform(player_df)
        else:
            player_features = player_df

        # 进行预测
        if hasattr(model, 'predict_proba'):
            risk_proba = model.predict_proba(player_features)[0, 1]
        else:
            risk_proba = float(model.predict(player_features)[0])

        risk_level = calculate_risk_level(risk_proba)

        # 计算特征贡献（如果模型支持）
        contributing_factors = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = getattr(model, 'feature_names_in_',
                                 [f'feature_{i}' for i in range(len(model.feature_importances_))])
            contributing_factors = dict(zip(feature_names, model.feature_importances_))

        # 生成建议
        recommendations = generate_recommendations(risk_proba, contributing_factors)

        return PredictionResponse(
            player_id=player_data.player_id,
            injury_risk=risk_proba,
            risk_level=risk_level,
            confidence=0.85,  # 实际应该根据模型的不确定性计算
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_data: BatchPlayerInput,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """批量球员伤病风险预测"""

    start_time = datetime.now()

    model_name = batch_data.model_name or "default"
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = loaded_models[model_name]
    predictions = []

    try:
        # 转换所有球员数据为DataFrame
        players_data = [player.dict() for player in batch_data.players]
        players_df = pd.DataFrame(players_data)

        # 批量数据验证
        if data_validator:
            validation_result = data_validator.validate_dataframe(players_df)
            if not validation_result['is_valid']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch data validation failed: {validation_result['errors']}"
                )

        # 批量特征工程
        if feature_engineer:
            players_features = feature_engineer.transform(players_df)
        else:
            players_features = players_df

        # 批量预测
        if hasattr(model, 'predict_proba'):
            risk_probas = model.predict_proba(players_features)[:, 1]
        else:
            risk_probas = model.predict(players_features)

        # 处理每个预测结果
        for i, (player, risk_proba) in enumerate(zip(batch_data.players, risk_probas)):
            risk_level = calculate_risk_level(risk_proba)

            # 计算特征贡献
            contributing_factors = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = getattr(model, 'feature_names_in_',
                                     [f'feature_{j}' for j in range(len(model.feature_importances_))])
                contributing_factors = dict(zip(feature_names, model.feature_importances_))

            recommendations = generate_recommendations(risk_proba, contributing_factors)

            prediction = PredictionResponse(
                player_id=player.player_id,
                injury_risk=float(risk_proba),
                risk_level=risk_level,
                confidence=0.85,
                contributing_factors=contributing_factors,
                recommendations=recommendations
            )
            predictions.append(prediction)

        # 生成汇总统计
        risk_levels = [p.risk_level for p in predictions]
        summary = {
            "total_players": len(predictions),
            "high_risk_count": risk_levels.count("HIGH"),
            "medium_risk_count": risk_levels.count("MEDIUM"),
            "low_risk_count": risk_levels.count("LOW"),
            "average_risk": float(np.mean([p.injury_risk for p in predictions])),
            "model_used": model_name
        }

        processing_time = (datetime.now() - start_time).total_seconds()

        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/upload/csv")
async def upload_csv_for_prediction(
    file: UploadFile = File(...),
    model_name: str = "default",
    user=Depends(get_current_user)
):
    """上传CSV文件进行批量预测"""

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    try:
        # 读取CSV文件
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))

        # 转换为PlayerInput列表
        players = []
        for _, row in df.iterrows():
            player_dict = row.to_dict()
            # 处理可能的缺失字段
            if 'injury_history' in player_dict and isinstance(player_dict['injury_history'], str):
                player_dict['injury_history'] = player_dict['injury_history'].split(',') if player_dict['injury_history'] else []

            player = PlayerInput(**player_dict)
            players.append(player)

        # 调用批量预测
        batch_input = BatchPlayerInput(players=players, model_name=model_name)
        return await predict_batch(batch_input, BackgroundTasks(), user)

    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")


@app.get("/download/results/{prediction_id}")
async def download_results(prediction_id: str):
    """下载预测结果"""
    # 这里应该实现结果存储和检索逻辑
    # 目前返回一个示例响应
    raise HTTPException(status_code=501, detail="Download functionality not implemented yet")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("Starting Sports Injury Risk Prediction API...")
    load_models()
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("Shutting down Sports Injury Risk Prediction API...")
    global loaded_models
    loaded_models.clear()
    logger.info("API shutdown complete")


# 异常处理
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )