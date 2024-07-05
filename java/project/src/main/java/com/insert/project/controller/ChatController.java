package com.insert.project.controller;

import com.insert.project.domain.entity.Member;
import com.insert.project.dto.BoardDTO;
import com.insert.project.dto.CommentDTO;
import com.insert.project.service.BoardService;
import com.insert.project.service.CommentService;
import com.insert.project.session.SessionConst;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;

@Controller
@RequiredArgsConstructor
@RequestMapping("/chat")
public class ChatController {
    private final BoardService boardService;
    private final CommentService commentService;

//    @GetMapping("")
//    public String paging(@PageableDefault(page = 1) Pageable pageable, Model model,
//                         @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
////        pageable.getPageNumber();
//        if (loginMember == null) {
//            return "/main/login";
//        }
//        model.addAttribute("member", loginMember);
//
//        Page<BoardDTO> boardList = boardService.paging(pageable);
//        int blockLimit = 3;
//        int startPage = (((int) (Math.ceil((double) pageable.getPageNumber() / blockLimit))) - 1) * blockLimit + 1; // 1 4 7 10 ~~
//        int endPage = ((startPage + blockLimit - 1) < boardList.getTotalPages()) ? startPage + blockLimit - 1 : boardList.getTotalPages();
//
//        // page 갯수 20개
//        // 현재 사용자가 3페이지
//        // 1 2 3
//        // 현재 사용자가 7페이지
//        // 7 8 9
//        // 보여지는 페이지 갯수 3개
//        // 총 페이지 갯수 8개
//
//        model.addAttribute("boardList", boardList);
//        model.addAttribute("startPage", startPage);
//        model.addAttribute("endPage", endPage);
//        return "/chat/list";
//    }


    @GetMapping("/save")
    public String saveForm(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model) {
        if (loginMember == null) {
            return "/main/login";
        }
        model.addAttribute("member", loginMember);
        return "chat/save";
    }


    @PostMapping("/save")
    public String save(@ModelAttribute BoardDTO boardDTO, Model model,
                       @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) throws IOException {
        if (loginMember == null) {
            return "/main/login";
        }
        model.addAttribute("member", loginMember);

        boardService.save(boardDTO, loginMember);
        return "redirect:/chat";
    }
    @GetMapping("/start")
    public String boardchat(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model){
        //세션에 회원 데이터가 없으면 home
        if (loginMember == null) {
            return "main/login";
        }
        //세션이 유지되면  이동
        model.addAttribute("member", loginMember);
        return "chat/start";
    }
    @GetMapping("/{id}")
    public String findById(@PathVariable Long id, Model model,
                           @PageableDefault(page=1) Pageable pageable,
                           @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);

        boardService.updateHits(id);
        BoardDTO boardDTO = boardService.findById(id);
        List<CommentDTO> commentDTOList = commentService.findAll(id);
        model.addAttribute("commentList", commentDTOList);
        model.addAttribute("board", boardDTO);
        model.addAttribute("page", pageable.getPageNumber());
        return "chat/detail";
    }

    @GetMapping("/map")
    public String boardmap(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model){
        //세션에 회원 데이터가 없으면 home
        if (loginMember == null) {
            return "main/login";
        }
        //세션이 유지되면  이동
        model.addAttribute("member", loginMember);
        return "chat/map";
    }

}
