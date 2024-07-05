package com.insert.project.controller;

import com.insert.project.domain.entity.BoardEntity;
import com.insert.project.domain.entity.Member;
import com.insert.project.service.BoardService;
import com.insert.project.service.PostService;
import com.insert.project.session.SessionConst;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.SessionAttribute;

import java.util.List;

@Controller
@RequiredArgsConstructor
public class ChatController2 {
   // @Autowired

    @Autowired
    private final BoardService boardService;
    private final PostService postService;





}



