#!/usr/bin/env python3
"""
EXIF图片筛选API服务
提供本地图片扫描、EXIF读取、时间范围筛选等功能
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from PIL.ExifTags import TAGS
import logging

app = Flask(__name__)
CORS(
    app,
    resources=r"/*",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff', '.heic', '.heif'}


def get_exif_data(image_path):
    """
    读取图片EXIF数据
    返回: {
        'DateTimeOriginal': 'YYYY:MM:DD HH:MM:SS',
        'Orientation': 1-8,
        'UserComment': '...',
        ...
    }
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif() if hasattr(img, '_getexif') else None
        
        if not exif_data:
            return {}
        
        result = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            result[tag_name] = value
        
        return result
    except Exception as e:
        logger.error(f"读取EXIF失败 {image_path}: {e}")
        return {}


def parse_exif_datetime(dt_str):
    """
    解析EXIF日期时间字符串 'YYYY:MM:DD HH:MM:SS' 为时间戳
    """
    if not dt_str:
        return None
    try:
        dt = datetime.strptime(str(dt_str), '%Y:%m:%d %H:%M:%S')
        return int(dt.timestamp() * 1000)  # 毫秒时间戳
    except Exception as e:
        logger.error(f"解析日期失败: {dt_str}, {e}")
        return None


def parse_user_comment(user_comment):
    """
    解析UserComment，判断是否包含 ConfirmedDate=1
    """
    if not user_comment:
        return False
    
    # 处理字节数组情况
    if isinstance(user_comment, bytes):
        try:
            user_comment = user_comment.decode('utf-8', errors='ignore')
        except:
            pass
    
    user_comment_str = str(user_comment)
    return 'ConfirmedDate=1' in user_comment_str


def image_to_base64(image_path):
    """
    将图片转换为base64编码
    """
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"图片转base64失败 {image_path}: {e}")
        return None


def scan_crops_directory(root_path, start_time_ms, end_time_ms):
    """
    扫描 root_path/*/crops/* 下的图片，按EXIF时间筛选
    返回: [
        {
            'filename': '文件名',
            'relPath': '相对路径',
            'base64': 'base64编码的图片',
            'exif': {
                'DateTimeOriginal': '...',
                'Orientation': 1,
                'UserComment': '...',
                ...
            },
            'isLocked': True/False
        },
        ...
    ]
    """
    results = []
    root = Path(root_path)
    
    if not root.exists():
        logger.error(f"路径不存在: {root_path}")
        return results
    
    # 遍历 root/*/crops/* 结构
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        
        crops_dir = subdir / 'crops'
        if not crops_dir.exists():
            continue
        
        for img_file in crops_dir.rglob('*'):
            if not img_file.is_file():
                continue
            
            # 检查扩展名
            if img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            
            # 读取EXIF
            exif_data = get_exif_data(str(img_file))
            dt_original = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')
            
            # 解析时间戳
            dt_ms = parse_exif_datetime(dt_original)
            if dt_ms is None:
                logger.warning(f"无法获取EXIF时间: {img_file}")
                continue
            
            # 时间范围筛选
            if dt_ms < start_time_ms or dt_ms > end_time_ms:
                continue
            
            # 读取图片为base64
            b64 = image_to_base64(str(img_file))
            if not b64:
                continue
            
            # 解析锁定状态
            user_comment = exif_data.get('UserComment', '')
            is_locked = parse_user_comment(user_comment)
            
            # 获取相对路径
            try:
                rel_path = str(img_file.relative_to(root)).replace('\\', '/')
            except:
                rel_path = str(img_file)
            
            # 获取Orientation
            orientation = exif_data.get('Orientation', 1)
            
            results.append({
                'filename': img_file.name,
                'relPath': rel_path,
                'base64': b64,
                'exif': {
                    'DateTimeOriginal': str(dt_original) if dt_original else '',
                    'Orientation': int(orientation) if orientation else 1,
                    'UserComment': str(user_comment) if user_comment else ''
                },
                'isLocked': is_locked
            })
    
    # 按相对路径排序
    results.sort(key=lambda x: x['relPath'])
    return results


@app.route('/api/scan-crops', methods=['POST'])
def scan_crops():
    """
    POST /api/scan-crops
    请求体:
    {
        "rootPath": "/path/to/root",
        "startTime": "2024-01-01T00:00",
        "endTime": "2024-12-31T23:59"
    }
    
    返回:
    {
        "success": true,
        "data": [...],
        "count": 10,
        "message": "..."
    }
    """
    try:
        payload = request.get_json()
        root_path = payload.get('rootPath', '')
        start_time_str = payload.get('startTime', '')
        end_time_str = payload.get('endTime', '')
        
        if not root_path:
            return jsonify({'success': False, 'message': '缺少rootPath参数'}), 400
        
        if not start_time_str or not end_time_str:
            return jsonify({'success': False, 'message': '缺少时间参数'}), 400
        
        # 解析时间
        try:
            start_dt = datetime.fromisoformat(start_time_str.replace('T', ' '))
            end_dt = datetime.fromisoformat(end_time_str.replace('T', ' '))
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
        except Exception as e:
            return jsonify({'success': False, 'message': f'时间格式错误: {e}'}), 400
        
        # 扫描目录
        results = scan_crops_directory(root_path, start_ms, end_ms)
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results),
            'message': f'成功加载 {len(results)} 张图片'
        })
    
    except Exception as e:
        logger.error(f"扫描失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok'})


@app.route('/', methods=['GET'])
def index():
    """返回fix.html文件"""
    return send_from_directory(os.path.dirname(__file__), 'fix.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=15000, debug=True)
