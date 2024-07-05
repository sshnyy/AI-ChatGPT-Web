package com.insert.project;

import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.logging.Logger;

public class Client {
    private final static Logger LOG = Logger.getGlobal();

    public static void txt2server(String input, Socket socket, WebSocketSession session) {
        try {

            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);

                // 메시지를 입력받아 서버로 전송

                out.println(input);

                String msg = in.readLine(); // 서버에서 메시지 수신 대기

                LOG.info("Server sent: " + msg);
                System.out.println("Server sent: " + msg);

                TextMessage ServerMessage = new TextMessage(msg);
                session.sendMessage(ServerMessage);


        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
