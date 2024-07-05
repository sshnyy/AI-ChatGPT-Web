package com.insert.project.controller;

import com.insert.project.domain.entity.Member;
import com.insert.project.domain.repository.MemberRepository;
import com.insert.project.session.SessionConst;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.SessionAttribute;

import javax.servlet.http.HttpServletRequest;

@Slf4j
@Controller
@RequiredArgsConstructor
public class HomeController {

    private final MemberRepository memberRepository;

    @GetMapping("/")
    public String homeLogin(
            @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model) {

        //세션에 회원 데이터가 없으면 home
        if (loginMember == null) {
            return "main/home.html";
        }

        //세션이 유지되면 로그인으로 이동
        model.addAttribute("member", loginMember);
        return "main/loginHome";
    }
    @GetMapping("/main/signup")

    public String boardsignup(){

        return "main/signup";
    }
    @GetMapping("/main/login")
    public String boardlogin(){

        return "main/login";
    }

    @GetMapping("/main/map")
    public String boardmao(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model){
        //세션에 회원 데이터가 없으면 home
        if (loginMember == null) {
            return "main/login";
        }
        //세션이 유지되면  이동
        model.addAttribute("member", loginMember);
        return "main/map";
    }


}