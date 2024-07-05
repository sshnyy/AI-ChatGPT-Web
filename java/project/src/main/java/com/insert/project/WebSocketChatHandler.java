package com.insert.project;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.net.Socket;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;


@Component
public class WebSocketChatHandler extends TextWebSocketHandler {

    private final static Logger LOG = Logger.getGlobal();
    private static boolean checkServerStar = false;

    static Socket socket;
    Client client = new Client();
    public static Queue<String> messageQueue = new LinkedList<>(); // 큐 생성

    private static final List<String> QUESTIONS = Arrays.asList(
            "피부의 위치와 증상을 채팅으로 정확히 알려주세요.",
            "증상이 얼마나 지속되었는지 알려주세요.",
            "병원에 내원한 적이 있나요?",
            "해당 피부 병변이 의심되는 부분의 이미지를 업로드해주세요.",
            "말씀하신 내용과 이미지를 바탕으로 정보를 제공해드리겠습니다. 아래의 제출 버튼을 누르면 1분이내로 답변을 받을 수 있습니다."
    );

    private final Map<WebSocketSession, Integer> sessionQuestionIndex = new ConcurrentHashMap<>();

    public static void startSocketIfServerStarted() {
        if (checkServerStar) {
            try {
                socket = new Socket("203.232.193.173", 5252);
                System.out.println("Connected to server.");
            } catch (IOException e) {
                System.out.println("Disconnected to server.");
                throw new RuntimeException(e);
            }
        }
    }

    // 소켓 통신 끊는 함수 추가
    public static void stopSocket() {
        if (socket != null && !socket.isClosed()) {
            try {
                socket.close();
                checkServerStar = false;
                System.out.println("Disconnected from server.");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    // 큐에서 메시지를 계속해서 처리하는 스레드를 생성하는 함수
    private void startMessageQueueProcessingThread(WebSocketSession session) {
        Thread thread = new Thread(() -> {
            while (true) {
                if (!messageQueue.isEmpty()) {
                    String input = messageQueue.poll();

                    if( input.equals("[PSTART]")){
                        // 파이썬 서버에 연결하고 시작
                        try{
                            checkServerStar = true;
                            startSocketIfServerStarted();
                            String start_msg = "[SERVER] Python Server Connected";
                            TextMessage ServerMessage = new TextMessage(start_msg);
                            session.sendMessage(ServerMessage);
                        }
                        catch (IOException e){
                            // 서버 시작 안될시 메세지 창만 올리기
                            String start_msg = "[SERVER] Python Server Disconnected";
                            TextMessage textMessage = new TextMessage(start_msg);
                            try {
                                session.sendMessage(textMessage);
                            } catch (IOException ex) {
                                throw new RuntimeException(ex);
                            }
                        }

                    } else if(checkServerStar && !input.equals("[PSTART]")){
                        // 서버 start 시에 파이썬 서버 연결
                        TextMessage textMessage = new TextMessage(input);
                        try {
                            session.sendMessage(textMessage);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        client.txt2server(input,socket,session);

                    } else if (checkServerStar && input.equals("[PSTOP]")){
                        TextMessage textMessage = new TextMessage(input);
                        try {
                            session.sendMessage(textMessage);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        stopSocket();
                    } else{
//                        // 서버 시작 안될시 메세지 창만 올리기
                        TextMessage textMessage = new TextMessage(input);
                        try {
                            session.sendMessage(textMessage);
                            if (input.equals("[1]")){
                                handleTextMessage(session,textMessage);
                            }
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                }

                try {
                    Thread.sleep(100); // 0.1-second wait before checking the queue again
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        thread.start();
    }


    private void sendNextQuestion(WebSocketSession session) throws IOException {
        int index = sessionQuestionIndex.getOrDefault(session, QUESTIONS.size());
        if (index < QUESTIONS.size()-1) {
            TextMessage question = new TextMessage(QUESTIONS.get(index));
            session.sendMessage(question);
        } else if (index == QUESTIONS.size()-1)  {
            TextMessage question = new TextMessage(QUESTIONS.get(index));
            session.sendMessage(question);
            sessionQuestionIndex.remove(session);
            // QUESTIONS 리스트의 내용을 모두 처리한 경우
            TextMessage completionMessage = new TextMessage("[2]");
            session.sendMessage(completionMessage);
        }
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        startMessageQueueProcessingThread(session); // 큐 처리 스레드 시작
        // 첫 번째 질문 전송 코드 제거
        super.afterConnectionEstablished(session);
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String input = message.getPayload();
        LOG.info("Received message: " + input);

        if (input.equals("[1]")) {
            startChatSession(session);
        } else if (input.startsWith("[IMG]")) {
            handleImageUpload(session, input.substring(5));
        } else if (sessionQuestionIndex.containsKey(session)) {
            handleUserResponse(session, input);
        } else {
            // 웹 소켓을 통해 받은 값이 "[1]"이 아니면 큐 처리를 하지 않음
            LOG.warning("Received unexpected message: " + input);
        }
    }

    private void startChatSession(WebSocketSession session) throws IOException {
        sessionQuestionIndex.put(session, 0);
        sendNextQuestion(session);
    }

    private void handleUserResponse(WebSocketSession session, String response) throws IOException {
        LOG.info("User response: " + response);
        sessionQuestionIndex.computeIfPresent(session, (s, idx) -> idx + 1);
        sendNextQuestion(session);
    }




    private void handleImageUpload(WebSocketSession session, String fileId) {
        // Implement your logic for handling image uploads here
        TextMessage responseMessage = new TextMessage("[SERVER] Image Upload Received: " + fileId);
        try {
            session.sendMessage(responseMessage);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        // WebSocket 연결이 종료되면 큐 처리 스레드도 종료합니다.
        // 해당 작업은 필요에 따라 구현해야 합니다.
        super.afterConnectionClosed(session, status);
    }
}
