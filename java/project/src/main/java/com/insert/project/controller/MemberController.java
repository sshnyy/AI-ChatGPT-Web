package com.insert.project.controller;


import com.insert.project.domain.entity.Member;
import com.insert.project.exception.DuplicatedIdEx;
import com.insert.project.domain.repository.MemberRepository;
import com.insert.project.service.MemberService;
import com.insert.project.session.SessionConst;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.validation.Valid;


@Controller
@RequiredArgsConstructor
@RequestMapping("/members")
public class MemberController {

    private final MemberRepository memberRepository;

    private final MemberService memberService;
    @GetMapping("/add")
    public String addForm(@ModelAttribute("member") Member member) {
        return "main/signup";
    }

    @PostMapping("/add")
    public String save(@Valid @ModelAttribute Member member, BindingResult bindingResult, Model model) {

        if (bindingResult.hasErrors()) {
            return "main/signup";
        }
        Member findMember = memberRepository.findByLoginId(member.getLoginId()).orElse(null);
        if (findMember != null) {

            model.addAttribute("errorMessage", "이미 존재하는 ID입니다.");
            //중복 ID 발생 시 다시 리디렉션
            return "main/signup";
        }

        memberRepository.save(member);
        return "redirect:/";
    }

    @GetMapping("/myPage")
    public String myPageGet(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model){
        model.addAttribute("member",loginMember);
        return "main/myPage";
    }

    @PostMapping("/myPage")
    public String UserUpdate(@ModelAttribute Member member, @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember
            , HttpServletRequest request){
        Member updateMember = memberService.update(member, loginMember);


        // 기존에 등록되어있던 회원에대한 세션정보는 그대로기 때문에
        // 수정된 회원 updateMember 정보를 세션에 다시 set 시켜 준다.
        HttpSession session = request.getSession(true);
        session.setAttribute(SessionConst.LOGIN_MEMBER, updateMember);
        return "redirect:/";
    }
}
