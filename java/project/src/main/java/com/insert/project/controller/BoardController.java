package com.insert.project.controller;

import com.insert.project.domain.entity.BoardEntity;
import com.insert.project.domain.entity.Member;
import com.insert.project.dto.BoardDTO;
import com.insert.project.dto.CommentDTO;
import com.insert.project.service.BoardService;
import com.insert.project.service.CommentService;
import com.insert.project.service.PostService;
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
@RequestMapping("/board")
public class BoardController {
    private final BoardService boardService;
    private final CommentService commentService;

    private final PostService postService;

    @GetMapping("/save")
    public String saveForm(@SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember, Model model) {
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);
        return "board/save";
    }
    @PostMapping("/save")
    public String save(@ModelAttribute BoardDTO boardDTO, Model model,
                       @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) throws IOException {
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);


        boardService.save(boardDTO,loginMember);
        return "redirect:/board/paging"; // 등록 후 목록 페이지로 리다이렉트
    }


    @GetMapping("/")
    public String findAll(Model model, @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);

        List<BoardEntity> findPosts = postService.findAllPostsExcludingTitle("chat_log");
        model.addAttribute("findPosts",findPosts);
        // List<BoardDTO> boardDTOList = boardService.findAll();
        // model.addAttribute("boardList", boardDTOList);
        // return "/board/list";
        return "main/posts";
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
        return "board/detail";
    }

    @GetMapping("/update/{id}")
    public String updateForm(@PathVariable Long id, Model model,
                             @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);

        BoardDTO boardDTO = boardService.findById(id);
        model.addAttribute("boardUpdate", boardDTO);
        return "board/update";
    }

    @PostMapping("/update/{id}")
    public String update(@ModelAttribute BoardDTO boardDTO, Model model,
                         @PathVariable Long id,
                         @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
//        pageable.getPageNumber();
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);

        // 게시글 아이디 Pathvariable방식으로 받아야함
        boardService.update(boardDTO,id); //<- 리턴타입 void라서 그냥 이렇게만 하면돼
        //model.addAttribute("board", board);
        // return "redirect:/detail";

        return "redirect:/board/{id}"; //페이지가 아니라 컨트롤러를 호출하네.

        //        return "redirect:/board/" + boardDTO.getId();
    }

    @GetMapping("/delete/{id}")
    public String delete(@PathVariable Long id){
        boardService.delete(id);
        return "redirect:/board/paging";
    }

    // /board/paging?page=1
    @GetMapping("/paging")
    public String paging(@PageableDefault(page = 1) Pageable pageable, Model model,
                         @SessionAttribute(name = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
//        pageable.getPageNumber();
        if (loginMember == null) {
            return "main/login";
        }
        model.addAttribute("member", loginMember);

        Page<BoardDTO> boardList = boardService.paging(pageable);
        int blockLimit = 10;
        int startPage = (((int)(Math.ceil((double)pageable.getPageNumber() / blockLimit))) - 1) * blockLimit + 1; // 1 4 7 10 ~~
        int endPage = ((startPage + blockLimit - 1) < boardList.getTotalPages()) ? startPage + blockLimit - 1 : boardList.getTotalPages();

        // page 갯수 20개
        // 현재 사용자가 3페이지
        // 1 2 3
        // 현재 사용자가 7페이지
        // 7 8 9
        // 보여지는 페이지 갯수 3개
        // 총 페이지 갯수 8개

        model.addAttribute("boardList", boardList);
        model.addAttribute("startPage", startPage);
        model.addAttribute("endPage", endPage);
        return "board/paging";

    }





}










