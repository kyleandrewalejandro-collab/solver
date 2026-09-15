# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.
import json
import os
import random
import re
import string
import time
import warnings
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import execjs
from loguru import logger
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import queue
import threading
from functools import lru_cache
from fake_useragent import UserAgent
import sys

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
warnings.filterwarnings("ignore", message=".*SIFT_create.*deprecated.*")

DEBUG = False

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

TOKEN_SERVER_URL = os.environ.get('TOKEN_SERVER_URL', 'https://cn31-web-atx-production.up.railway.app')
TOKEN_SAVE_ENDPOINT = f"{TOKEN_SERVER_URL}/api/save-token"

def send_token_to_server(token):
    try:
        payload = {"token": token}
        r = requests.post(TOKEN_SAVE_ENDPOINT, json=payload, timeout=5)
        return r.status_code in [200, 201]
    except:
        return False

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def emergency_fallback():

    return [(80, 70), (160, 120), (240, 90)]

def safe_list_access(lst, index, default=None):

    try:
        if lst is None or not isinstance(lst, (list, tuple)):
            return default
        if not (0 <= index < len(lst)):
            return default
        return lst[index]
    except:
        return default

# Mini class definition -- reconstructed for torch.load deserialization
# The model was saved with a Mini class definition in training __main__.
# We define it here so torch.load() can deserialize without AttributeError.
class Mini(nn.Module):
    """YOLO-style Mini detector network for captcha image inference.
    
    Deserialization-only definition. Forward pass not required for inference mode.
    The actual weights and architecture are loaded from the saved state dict.
    """
    def __init__(self):
        super(Mini, self).__init__()
        # Minimal init - actual architecture comes from state_dict
        pass
    
    def forward(self, x):
        # Not called during inference if using eval mode
        raise NotImplementedError("Use model.eval() and inference through loaded state")

_model_state = None
_model_lock = threading.Lock()

def initialize_global_model():
    global _model_state

    if _model_state is not None:
        return _model_state

    with _model_lock:
        if _model_state is not None:
            return _model_state

        # Register Mini in sys.modules so unpickler finds it during load
        sys.modules[__name__].Mini = Mini

        model_path = os.path.join(DIR_PATH, 'net.pkl')
        if not os.path.exists(model_path):
            logger.error("Model file net.pkl not found")
            return None

        try:
            # Now torch.load can deserialize Mini instances
            state = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=False)

            if 'net' in state:
                state['net'] = state['net'].to(DEVICE)
                state['net'].eval()

                if USE_CUDA:
                    state['net'] = state['net'].half()

            _model_state = state
            logger.success(f"Model loaded on {DEVICE}")
            return _model_state

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def get_global_model():
    global _model_state
    if _model_state is None:
        return initialize_global_model()
    return _model_state

@lru_cache(maxsize=5)
def get_compiled_js_cached(file_name):
    try:
        js_path = os.path.join(DIR_PATH, file_name)
        with open(js_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
        ctx = execjs.compile(js_code)
        return ctx
    except:
        return None

def get_compiled_js(file_name):
    return get_compiled_js_cached(file_name)

_sift_detector = None
_sift_lock = threading.Lock()

def get_sift_detector():
    global _sift_detector
    if _sift_detector is None:
        with _sift_lock:
            if _sift_detector is None:
                try:
                    _sift_detector = cv2.SIFT_create(nfeatures=50, contrastThreshold=0.08)
                    logger.info("Using cv2.SIFT_create()")
                except AttributeError:
                    try:
                        _sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=50, contrastThreshold=0.08)
                        logger.info("Using cv2.xfeatures2d.SIFT_create()")
                    except AttributeError:
                        logger.warning("SIFT not available, falling back to ORB")
                        _sift_detector = cv2.ORB_create(nfeatures=50)
    return _sift_detector

file_lock = threading.Lock()
TOKEN_OUTPUT_FILE = os.path.join(DIR_PATH, 'validated_tokens.txt')                         

REFERER = "https://mtacc.mobilelegends.com/"
ID = "fef5c67c39074e9d845f4bf579cc07af"
FP_H = "mtacc.mobilelegends.com"

DUN163_DOMAINS = [
    "https://c.dun.163.com",
    "https://c.dun.163yun.com"
]

def rotate_about_center(src, angle, scale=1.):
    try:
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LINEAR)
    except:
        return src

def parse_y_pred(ypred, anchors, class_types, islist=False, threshold=0.2, nms_threshold=0):

    try:
        if not anchors or not class_types:
            return [] if islist else None

        ceillen = 5 + len(class_types)
        sigmoid = lambda x: 1/(1+math.exp(-x))
        infos = []

        for idx in range(min(len(anchors), 3)):                          
            try:
                tensor_idx = 4 + idx * ceillen
                if tensor_idx >= ypred.shape[3]:
                    continue

                if USE_CUDA:
                    a = ypred[:,:,:,tensor_idx].cpu().detach().numpy()
                else:
                    a = ypred[:,:,:,tensor_idx].detach().numpy()

                for ii, i in enumerate(a[0]):
                    for jj, j in enumerate(i):
                        infos.append((ii, jj, idx, sigmoid(j)))
            except:
                continue

        if not infos:
            return [] if islist else None

        infos = sorted(infos, key=lambda i: -i[3])

        def get_xyxy_clz_con_emergency(info):

            try:
                gap = 416/ypred.shape[1]
                x, y, idx, con = info

                if idx >= len(anchors):
                    return None

                gp = idx * ceillen

                if (gp + 5 + len(class_types)) > ypred.shape[3]:
                    return None

                contain = torch.sigmoid(ypred[0, x, y, gp+4])
                pred_xy = torch.sigmoid(ypred[0, x, y, gp+0:gp+2])
                pred_wh = ypred[0, x, y, gp+2:gp+4]
                pred_clz = ypred[0, x, y, gp+5:gp+5+len(class_types)]

                if USE_CUDA:
                    pred_xy = pred_xy.cpu().detach().numpy()
                    pred_wh = pred_wh.cpu().detach().numpy()
                    pred_clz = pred_clz.cpu().detach().numpy()
                else:
                    pred_xy = pred_xy.detach().numpy()
                    pred_wh = pred_wh.detach().numpy()
                    pred_clz = pred_clz.detach().numpy()

                exp = math.exp
                cx, cy = float(pred_xy[0]), float(pred_xy[1])
                rx, ry = (cx + x)*gap, (cy + y)*gap

                pw, ph = float(pred_wh[0]), float(pred_wh[1])
                rw, rh = math.exp(pw) * safe_list_access(anchors, idx, (1, 1))[0], math.exp(ph) * safe_list_access(anchors, idx, (1, 1))[1]

                x1, y1 = int(rx - rw/2), int(ry - rh/2)
                x2, y2 = int(rx + rw/2), int(ry + rh/2)

                clz_max_idx = int(np.argmax(pred_clz))
                clz_max_con = float(pred_clz[clz_max_idx])

                return (x1, y1, x2, y2, clz_max_con, contain.cpu().item() if USE_CUDA else contain.item(), clz_max_idx)
            except:
                return None

        res = []
        for info in infos:
            d = get_xyxy_clz_con_emergency(info)
            if d:
                res.append(d)
                if len(res) >= 15:
                    break

        return res if islist else {'result': res}

    except Exception as e:
        return [] if islist else None


class Dun163(object):

    def __init__(self, id_, referer, fp_h, ua, thread_id=1, domain="https://c.dun.163.com"):
        self.id = id_
        self.referer = referer
        self.fp_h = fp_h
        self.ua = ua
        self.thread_id = thread_id
        self.domain = domain
        self.ss = self.set_session()
        self.ctx = get_compiled_js('dun163.js')
        self.fp = self.get_fp()
        self.resp_json2 = None

    def get_fp(self):
        try:
            ctx = get_compiled_js('fp.js')
            if ctx:
                fp = ctx.call('fp', self.ua)
                return fp
            return 'fp_error'
        except:
            return 'fp_error'

    def set_session(self):
        ss = requests.Session()
        ss.headers.update({
            'User-Agent': self.ua,
            'Referer': self.referer
        })
        return ss

    def request_getconf(self):
        try:
            url = f"{self.domain}/api/security/getconf"
            params = {
                'id': self.id,
                'fp': self.fp,
                'referer': self.referer,
                't': int(time.time() * 1000)
            }

            resp = self.ss.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except:
            return None

    def request_get(self, dt, bid, ac_token=None, ir_token=None):
        try:
            url = f"{self.domain}/api/security/get"
            data = {
                'id': self.id,
                'fp': self.fp,
                'referer': self.referer,
                'dt': dt,
                'bid': bid,
                't': int(time.time() * 1000)
            }

            if ac_token:
                data['ac'] = ac_token
            if ir_token:
                data['ir'] = ir_token

            resp = self.ss.post(url, json=data, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except:
            return None

    def request_check(self, dt, bid, token='', captcha_type=7, click_data=None):
        try:
            url = f"{self.domain}/api/security/check"

            if self.ctx is None:
                js_time = 0
            else:
                t_start = time.time()
                try:
                    yidun_token = self.ctx.call('token', token, self.fp, dt, click_data)
                except:
                    yidun_token = ''
                js_time = int((time.time() - t_start) * 1000)

            data = {
                'id': self.id,
                'fp': self.fp,
                'referer': self.referer,
                'dt': dt,
                'bid': bid,
                'captcha_type': captcha_type,
                'token': yidun_token,
                't': int(time.time() * 1000)
            }

            resp = self.ss.post(url, json=data, timeout=10)
            if resp.status_code == 200:
                return resp.json(), js_time
            return None, js_time
        except Exception as e:
            return None, 0

    def handle_click_captcha_hybrid(self, bg_url, token, attempt_num=0):
        try:
            img_time_start = time.time()

            try:
                resp = self.ss.get(bg_url, timeout=10)
                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except:
                logger.warning(f"T-{self.thread_id} | Failed to download image, using fallback")
                return self.generate_random_click_points(), 0

            if img is None or img.size == 0:
                logger.warning(f"T-{self.thread_id} | Invalid image, using fallback")
                return self.generate_random_click_points(), 0

            img_time = int((time.time() - img_time_start) * 1000)

            logger.info(f"T-{self.thread_id} | Image size: {img.shape}")

            click_points = self.infer_click_points(img, attempt_num)

            if click_points and len(click_points) > 0:
                logger.info(f"T-{self.thread_id} | Inferred {len(click_points)} click points")
                return click_points, img_time

            logger.warning(f"T-{self.thread_id} | Inference failed, using fallback")
            return self.generate_random_click_points(), img_time

        except Exception as e:
            logger.error(f"T-{self.thread_id} | handle_click_captcha error: {e}")
            return self.generate_random_click_points(), 0

    def infer_click_points(self, img, attempt_num=0):
        try:
            model_state = get_global_model()
            if not model_state or 'net' not in model_state:
                logger.warning(f"T-{self.thread_id} | Model not available")
                return None

            model = model_state['net']

            img_h, img_w = img.shape[:2]

            target_h, target_w = 416, 416
            scale = min(target_w / img_w, target_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)

            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            offset_y = (target_h - new_h) // 2
            offset_x = (target_w - new_w) // 2
            canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img_resized

            img_tensor = torch.from_numpy(canvas).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            if USE_CUDA:
                img_tensor = img_tensor.to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor)

            anchors = [
                [(80, 70), (160, 120), (240, 90)],
                [(80, 70), (160, 120), (240, 90)],
                [(80, 70), (160, 120), (240, 90)],
            ]
            class_types = ['target']

            predictions = parse_y_pred(output, anchors, class_types, islist=True, threshold=0.2)

            if not predictions:
                logger.warning(f"T-{self.thread_id} | No predictions")
                return None

            scale_x = img_w / new_w if new_w > 0 else 1
            scale_y = img_h / new_h if new_h > 0 else 1

            click_points = []
            for pred in predictions[:5]:
                x1, y1, x2, y2, clz_con, obj_con, clz_idx = pred
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                orig_cx = int((cx - offset_x) * scale_x)
                orig_cy = int((cy - offset_y) * scale_y)

                orig_cx = max(0, min(orig_cx, img_w - 1))
                orig_cy = max(0, min(orig_cy, img_h - 1))

                click_points.append({
                    "x": orig_cx,
                    "y": orig_cy
                })

            return click_points if len(click_points) > 0 else None

        except Exception as e:
            logger.error(f"T-{self.thread_id} | infer_click_points error: {e}")
            return None

    def generate_random_click_points(self):
        try:
            patterns = [
                [(80, 70), (160, 120), (240, 90)],
                [(90, 80), (170, 110), (250, 100)],
                [(70, 90), (150, 140), (230, 80)],
                [(100, 60), (180, 130), (260, 110)],
                [(85, 75), (165, 115), (245, 95)],
                [(75, 85), (155, 125), (235, 85)],
                [(95, 65), (175, 135), (255, 105)],
                [(80, 90), (160, 110), (240, 100)],
                [(100, 70), (180, 120), (260, 90)],
                [(160, 60), (110, 130), (210, 140)],
            ]

            selected_pattern = random.choice(patterns)
            click_points = []

            for x, y in selected_pattern:
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)

                final_x = max(15, min(x + offset_x, 305))
                final_y = max(15, min(y + offset_y, 185))

                click_points.append({
                    "x": final_x,
                    "y": final_y
                })

            return click_points
        except:
            return [{"x": 80, "y": 70}, {"x": 160, "y": 120}, {"x": 240, "y": 90}]

    def save_token_locally(self, validate_token):
        try:
            line = f"{validate_token}\n"
            with file_lock:
                with open(TOKEN_OUTPUT_FILE, 'a') as f:
                    f.write(line)
            return True
        except Exception as e:
            logger.error(f"T-{self.thread_id} | Local save error: {e}")
            return False

    def run(self, attempt_num=0):

        try:

            if attempt_num > 0 and attempt_num % 5 == 0:
                logger.info(f"T-{self.thread_id} | Refreshing session...")
                self.ss = self.set_session()
                self.ctx = get_compiled_js('dun163.js')

            get_conf_data = self.request_getconf()
            if not get_conf_data:
                return False

            dt = get_conf_data.get('dt')
            ac_data = get_conf_data.get('ac', {})
            ac_token = ac_data.get('token')
            bid = ac_data.get('bid')

            ir_data = get_conf_data.get('ir', {})
            ir_token = ir_data.get('token') if ir_data.get('enable') else None

            get_data = self.request_get(dt, bid, ac_token, ir_token)
            if not get_data:
                return False

            captcha_type = get_data.get('type', 7)
            token = get_data.get('token')

            if not token:
                return False

            if captcha_type == 7:
                bg_urls = get_data.get('bg', [])
                if not bg_urls:
                    return False

                click_points, img_time = self.handle_click_captcha_hybrid(bg_urls[0], token, attempt_num)
                resp_json, js_time = self.request_check(dt, bid, token=token, captcha_type=7, click_data=click_points)
            else:
                return False

            self.resp_json2 = resp_json

            if resp_json.get('result') == True:
                validate_raw = resp_json.get('validate', '')
                validate_decoded = ""

                if validate_raw and self.ctx:
                    try:
                        validate_decoded = self.ctx.call('do_onVerify', validate_raw, self.fp)
                    except Exception as e:
                        logger.error(f"T-{self.thread_id} | JS decode error: {e}")
                        return False

                if validate_decoded and len(validate_decoded.strip()) > 10:
                    server_success = send_token_to_server(validate_decoded)
                    if server_success:
                        logger.success(f'T-{self.thread_id} SUCCESS: {validate_decoded[:40]}... | Sent to server')
                    else:
                        self.save_token_locally(validate_decoded)
                        logger.success(f'T-{self.thread_id} SUCCESS: {validate_decoded[:40]}... | Saved locally')
                    return True
                else:
                    logger.warning(f'T-{self.thread_id} | Invalid token: {validate_decoded}')
                    return True
            else:

                error_msg = resp_json.get('msg', 'Unknown error')
                logger.warning(f'T-{self.thread_id} | Verification failed: {error_msg}, resetting session...')
                self.ss = self.set_session()
                return False

        except Exception as e:
            logger.error(f'T-{self.thread_id} | run() failed: {str(e)[:100]}')

            try:
                self.ss = self.set_session()
            except:
                pass
            return False

def worker_thread(thread_id, config):

    while True:                                           
        try:
            logger.info(f"T-{thread_id} | Creating new solver instance...")

            d = Dun163(
                id_=config['ID_'], 
                referer=config['REFERER'], 
                fp_h=config['FP_H'], 
                ua=config['UA'], 
                thread_id=thread_id, 
                domain=config['DOMAIN']
            )

            attempt = 0
            success_count = 0
            consecutive_failures = 0

            while True:                                  
                attempt += 1

                if consecutive_failures > 10:
                    logger.warning(f"T-{thread_id} | Too many failures ({consecutive_failures}), recreating solver...")
                    break                                              

                try:

                    time.sleep(random.uniform(1.0, 3.0))

                    success = d.run(attempt_num=attempt)

                    if success:
                        success_count += 1
                        consecutive_failures = 0
                        logger.info(f"T-{thread_id} | Attempt {attempt} | Success #{success_count}")

                        time.sleep(random.uniform(0.5, 1.0))
                    else:
                        consecutive_failures += 1
                        logger.warning(f"T-{thread_id} | Attempt {attempt} | Failed ({consecutive_failures} in a row)")

                        if consecutive_failures > 5:
                            wait_time = min(consecutive_failures * 2, 30)
                            logger.info(f"T-{thread_id} | Backing off for {wait_time}s")
                            time.sleep(wait_time)

                except Exception as e:
                    logger.error(f"T-{thread_id} | Run error: {e}")
                    consecutive_failures += 1
                    time.sleep(5)

            logger.info(f"T-{thread_id} | Recreating solver...")
            continue

        except Exception as e:
            logger.error(f"T-{thread_id} | Worker crashed: {e}")
            logger.info(f"T-{thread_id} | Restarting in 10 seconds...")
            time.sleep(10)
            continue                           
def main():
    logger.info("Starting CN31 Solver...")

    model_state = initialize_global_model()
    if not model_state:
        logger.error("Model not available - cannot continue")
        return

    js_ctx = get_compiled_js('dun163.js')
    if not js_ctx:
        logger.error("JavaScript not available - cannot continue")
        return

    sift_detector = get_sift_detector()
    logger.success("All resources loaded")

    config = {
        'ID_': ID,
        'REFERER': REFERER,
        'FP_H': FP_H,
        'UA': UserAgent().random,
        'DOMAIN': DUN163_DOMAINS[0]
    }

    NUM_THREADS = 3                                  

    logger.info(f"Starting {NUM_THREADS} worker threads")
    logger.info(f"ID: {ID}")
    logger.info(f"REFERER: {REFERER}")
    logger.info(f"Server URL: {TOKEN_SERVER_URL}")
    logger.info("-" * 50)

    while True:
        try:
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = []

                for i in range(NUM_THREADS):
                    thread_config = config.copy()
                    thread_config['UA'] = UserAgent().random
                    thread_config['DOMAIN'] = DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
                    future = executor.submit(worker_thread, i+1, thread_config)
                    futures.append(future)

                for future in futures:
                    future.result()

        except KeyboardInterrupt:
            logger.warning("Stopping...")
            executor.shutdown(wait=True)
            break

        except Exception as e:
            logger.error(f"Main loop crashed: {e}")
            logger.info("Restarting main loop in 10 seconds...")
            time.sleep(10)
if __name__ == '__main__':
    main()