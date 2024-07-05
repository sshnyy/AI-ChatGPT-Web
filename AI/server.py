import socket
import threading
import pymysql
from pipeline import pipe
from kfold_pipeline import kfold_pipe
from utils.common.fix_seed import seed_everything
from utils.common.constant import LABEL_DICT
from utils.common.translation import str2bool
import argparse
import openai
from datetime import datetime
from bs4 import BeautifulSoup

HOST = 'your ip address'
PORT = 5252

def chatgpt_access(prediction,board_contents, board_title):
    # 모델 - GPT 3.5 Turbo 선택
    model = "gpt-3.5-turbo"

    # 질문 작성하기
    query = f"회원의 피부 이미지 예측 결과 {prediction} 입니다.\n 회원의 질문 제목: {board_title} \n 회원의 질문 내용: {board_contents}"

    # 메시지 설정하기
    messages = [
            {"role": "system", "content": "당신은 피부병변에 대해 진단하는 챗봇입니다. 사용자의 질문은 게시글 형식으로 주어지며, 이미지에 대한 예측 결과와 게시글 제목, 내용을 받습니다. 해당 피부병변에 대해 주어지는 정보들을 참고하여 답변하세요."},
            {"role": "user", "content": query}
    ]

    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature = 0.3
    )
    answer = response['choices'][0]['message']['content']
    return answer

def fetch_image_path(img_id, cursor):
    query = "SELECT stored_file_name FROM board_file_table WHERE board_id = %s"
    cursor.execute(query, (img_id,))
    result = cursor.fetchone()
    return "/home/suyeon/code/capstone3/DL/image/" + result[0] if result else None

def fetch_board_content_title(img_id, cursor):
    query = "SELECT board_contents, board_title FROM board_table WHERE board_id = %s"
    cursor.execute(query, (img_id,))
    return cursor.fetchone()

def handle_client(args, conn, addr):
    print('Connected by', addr)
    with conn:
        while True:
            data = conn.recv(1024).decode('utf-8').strip()
            if not data or data == '[PSTOP]':
                print('Closing connection with', addr)
                break

            print('Client sent:', data)

            if 'IMG' in data:
                try:
                    img_id = int(data.split()[1])
                    db = pymysql.connect(user='user', password='pwd', host='input your ip address', port=3306, db='insertDB', charset='utf8')
                    cursor = db.cursor()

                    img_path = fetch_image_path(img_id, cursor)
                    if img_path:
                        print(f"Found image path: {img_path}")

                        LABEL_IDX = {i:v for v,i in LABEL_DICT.items()}
                        prediction = LABEL_IDX[pipe(args, args.device, img_path) if args.num_folds == 1 else kfold_pipe(args, args.device, img_path)]
                        print(f"Prediction: {prediction}")
                        conn.sendall(f'[SERVER] Prediction: {prediction}\n'.encode())
                        
                        board_contents, board_title = fetch_board_content_title(img_id, cursor)
                        # board_contents에 들어있는 HTML 형식의 내용
                        html_content = board_contents
                        # BeautifulSoup을 사용하여 텍스트 추출
                        soup = BeautifulSoup(html_content, 'html.parser')
                        text_content = soup.get_text()
                        print(f"Board Title: {board_title}")
                        print(f"Board Contents: {text_content}")

                        answer = chatgpt_access(prediction,text_content, board_title)
                        print(f"answer: ", answer)
                        # DB에 답변 등록
                        # 현재 시간 가져오기
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        
                        # 데이터베이스에 레코드 추가
                        insert_query = """
                        INSERT INTO comment_table (board_id, comment_contents, comment_writer, created_time)
                        VALUES (%s, %s, %s, %s)
                        """
                        cursor.execute(insert_query, (img_id, answer, "Chat-GPT", current_time))
                        db.commit()


                    else:
                        print("Image ID not found")

                except pymysql.MySQLError as err:
                    print(f"Error: {err}")


                finally:
                    cursor.close()
                    db.close()

            else:
                response = "HI"
                conn.sendall(f'[SERVER] {response}\n'.encode())

def start_server(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        print(f'Server listening on {HOST}:{PORT}')
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(args, conn, addr)).start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--OPENAI_API_KEY", type=str, default= "")
    parser.add_argument("--save_path", type=str, default="./models/saved_model/")
    parser.add_argument("--model_saved_path", type=str, default="./models/saved_model/resnet_50d/f1/f1_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--kfold_model_saved_path",
        type=str,
        default="./models/saved_model/resnet_50d/f1/*/f1_best.pt",
    )
    parser.add_argument("--backbone", type=str, default="resnet50d")
    parser.add_argument(
        "--tta", type=str2bool, default="False", help="test time augmentation"
    )
    parser.add_argument("--num_classes", type=str, default=len(LABEL_DICT))
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--predict_mask", type=str2bool ,default="False") # 예측시 입력받은 이미지에 대한 마스크 생성 여부
    parser.add_argument("--unet-checkpoint", type=str, default="/home/suyeon/code/capstone3/DL/models/unet/model/model_checkpoint.ckpt")
    args = parser.parse_args()
    openai.api_key = args.OPENAI_API_KEY
    start_server(args)
