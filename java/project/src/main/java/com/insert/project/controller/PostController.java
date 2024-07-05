package com.insert.project.controller;

import com.insert.project.domain.entity.BoardEntity;
import com.insert.project.domain.entity.Member;
import com.insert.project.domain.repository.PostRepository;
import com.insert.project.service.PostService;
import com.insert.project.session.SessionConst;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.SessionAttribute;

import java.util.List;

@Controller
@RequiredArgsConstructor
public class PostController {

    private final PostRepository postRepository;
    private final PostService postService;


    // 내가 만든 글 조회 -> GET
    @GetMapping("/myPage/post")
    public String myPageGet(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model){
        List<BoardEntity> findPosts = postService.findBoardsExcludingTitle(loginMember.getId(),"chat_log");
        model.addAttribute("findPosts",findPosts);
        return "main/myPosts";
    }

    @GetMapping("/chat")
    public String findAll(Model model, @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);

        List<BoardEntity> findPosts = postService.findMyChatLogPosts(loginMember.getId());
        model.addAttribute("findPosts",findPosts);
        return "chat/list";
    }


}

// 오늘 한거.
// DB 관계설정 -> 보드 생성할때 Member 주입(set)
// Controller -> Service -> Repository -> JPQL -> DB조회