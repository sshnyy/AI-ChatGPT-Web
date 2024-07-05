const chatContainer = document.getElementById("chatWindow");
const chatMessages = document.getElementById("plusMsg");
const options = document.getElementById("options");
const messageInput = document.getElementById("messageInput");

// 지도보기
function openMap() {
    window.open('map', 'Map Window', 'width=800, height=600');
}

// 웹소켓 관련 코드
var webSocket = new WebSocket("ws://" + location.host + "/ws/chat");

webSocket.onopen = function (event) {
    document.getElementById('websocketStatus').innerText = "Connected to Server";

};
webSocket.onmessage = function (event) {
    var message = event.data;
    var formattedDate = getTimeStamp();

    // 메시지가 "[2]"인 경우 "DB_upload" 버튼을 보이도록 변경
    if (message === "[2]") {
        document.getElementById('db_btn').style.display = 'block';
    } else {
        document.getElementById('db_btn').style.display = 'none';
        addMessageToChat(message, "left", formattedDate);
    }


};



webSocket.onclose = function(message) {
 document.getElementById('websocketStatus').innerText = "Server Disconnect...";
};

// getTimeStamp() 함수 (이 부분은 그대로 두세요)
function getTimeStamp() {
    var date = new Date();
    var options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: 'numeric', second: 'numeric', hour12: true };
    var formattedDate = date.toLocaleString('en-US', options);
    return formattedDate;
}

// JSON 파일을 불러오는 함수
function loadChatData() {
    fetch('/chat_flow.json')
        .then(response => response.json())
        .then(data => {
            // JSON 데이터를 받아온 후 대화 처리 로직을 호출
            handleChatData(data);
        })
        .catch(error => {
            console.error('JSON 파일을 불러오는 중 오류가 발생했습니다.', error);
        });
}

// 대화 처리 로직
function handleChatData(chatData) {
    let currentFlow = chatData.initial_prompt;
    let userInputInProgress = false; // 사용자 입력 진행 여부를 나타내는 변수

    function addMessage(message) {
        const messageDiv = document.createElement("div");
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
    }

    function displayOptions(optionsList) {
        options.innerHTML = "";
        optionsList.forEach(option => {
            const button = document.createElement("button");
            button.type = "button"; // type을 "button"으로 설정
            button.textContent = option.label;
            button.addEventListener("click", (event) => {
                event.preventDefault(); // 이벤트 기본 동작 막기
                handleOptionClick(option.action, option.label);
            });
            options.appendChild(button);
        });
    }

function handleOptionClick(action, label) {
    const nextFlow = chatData.flows[action];
    document.getElementById('db_btn').style.display = 'none';
    if (nextFlow) {
        if (nextFlow.type === "flow") {
            addMessageToChat("[USER] " + label, 'right'); // 버튼 레이블을 먼저 추가
            addMessageToChat(nextFlow.message, 'left'); // 다음 메시지 출력
            displayOptions(nextFlow.options); // 다음 선택지 표시
        } else if (nextFlow.type === "message") {
            addMessageToChat("[USER] " + label, 'right'); // 버튼 레이블을 먼저 추가
            addMessageToChat(nextFlow.content, 'left'); // 다음 메시지 출력
            if (nextFlow.options) {
                displayOptions(nextFlow.options); // 다음 선택지 표시
            } else {
                // 더 이상 나타낼 옵션이 없으면 "end_prompt"로 이동
                currentFlow = chatData.flows.end_prompt;
                addMessageToChat(currentFlow.message, 'left');
                displayOptions(currentFlow.options);
            }
        }
        currentFlow = nextFlow;
    } else if (action === "A" || action === "B" || action === "C") {
        // "또 다른 서비스를 이용하시겠습니까?"에서 A, B, C 버튼을 클릭한 경우
        currentFlow = chatData.end_prompt;
        addMessageToChat(currentFlow.message, 'left');
        displayOptions(currentFlow.options);
    }
}

// 초기 메시지와 선택지 표시
addMessageToChat(currentFlow.message, 'left');
displayOptions(currentFlow.options);
}

// 사용자가 입력한 메시지를 전송하는 함수
function sendMessage(event) {
    const userMessage = messageInput.value;
    event.preventDefault(); // 이벤트 기본 동작 막기
    if (userMessage) {
        webSocket.send(userMessage);
        addMessageToChat("[USER] " + userMessage, "right");
        messageInput.value = ""; // 입력 창 비우기

        // 여기에서 서버로 메시지를 전송하는 로직을 추가할 수 있습니다.
        // 예를 들어, WebSocket을 사용하여 서버에 메시지를 전송할 수 있습니다.
    }
    else {
            alert("메시지를 입력해주세요!"); // 사용자에게 메시지 입력을 요청
        }
}

// 메시지를 채팅창에 추가하는 함수
function addMessageToChat(message, direction) {
    const formattedDate = getTimeStamp();

    if (message ==="위치 주변 병원 보기"){
        openMap();

    }
    else if (message === "이용하기"){
        webSocket.send("[1]");
    }
    else {
        const messageDiv = document.createElement("div");
        messageDiv.className = "chat-bubble " + (direction === "left" ? "chat-bubble-left" : "chat-bubble-right");
        const messageContent = message.replace(/\n/g, "<br>"); // '\n'을 <br>로 변환
        messageDiv.innerHTML = messageContent + '<div class="chat-time">' + formattedDate + '</div>';
        chatMessages.appendChild(messageDiv);
        }
}

// 이미지를 채팅창에 추가하는 함수
function sendImage(inputElement) {
    var file = inputElement.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            var chatMessages = document.getElementById('plusMsg');
            var formattedDate = getTimeStamp();
            chatMessages.innerHTML += '<div class="chat-bubble chat-bubble-right"><img src="' + e.target.result + '" style="max-width:100%;"><div class="chat-time">' + formattedDate + '</div></div>';
            webSocket.send(" ");
            // 이 부분에서 WebSocket을 사용하여 이미지 데이터를 서버에 전송할 수 있습니다.
            // 예를 들면, webSocket.send(e.target.result);
            // 하지만 이미지는 크기가 크므로, Base64 인코딩 된 데이터를 직접 전송하는 것보다
            // 별도의 업로드 방식을 사용하는 것이 좋습니다.
        };
        reader.readAsDataURL(file);
    }
    }


 // DB에 업로드
function collectChatLog() {
    // 모든 채팅 메시지를 가져옴
    var chatHTML = '';
    var chatElements = document.querySelectorAll(".chat-bubble");
    chatElements.forEach(function(element) {
        var clone = element.cloneNode(true);
        var imgs = clone.querySelectorAll("img");
        imgs.forEach(function(img) {
            var src = img.getAttribute("src");
            var placeholder = document.createElement("div");
            placeholder.classList.add("chat-image");
            img.parentNode.replaceChild(placeholder, img);
        });
        chatHTML += clone.outerHTML + "\n";
    });

    // boardContents textarea에 채팅 로그 저장
    document.getElementById('boardContents').value = chatHTML;
}
// 채팅 로그를 수집하고 폼을 제출하는 함수 추가
function collectChatLogAndSubmit() {
  collectChatLog();
  var form = document.querySelector("form.needs-validation");
  form.submit();
}


// 페이지 로드 시 JSON 데이터 로드 시작
loadChatData();