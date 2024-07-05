package com.insert.project.service;

import com.insert.project.domain.entity.BoardEntity;
import com.insert.project.domain.repository.PostRepository;
import com.insert.project.dto.BoardDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class PostService {
    private final PostRepository postRepository;

    public List<BoardEntity> findBoardsExcludingTitle(Long memberId, String titleToExclude) {
        return postRepository.findByNameExcludingTitle(memberId, titleToExclude);
    }
    //return postRepository.findByName(memberId);

    public List<BoardEntity> findAllPostsExcludingTitle(String titleToExclude){

        return postRepository.findAllPostsExcludingTitle(titleToExclude);
    }


    public List<BoardEntity> findBoardsExcludingChat(Long memberId, String titleToExclude) {
        return postRepository.findByNameExcludingTitle(memberId, titleToExclude);
    }

    public List<BoardEntity>findMyChatLogPosts(Long memberId){
        return postRepository.findMyChatLogPosts(memberId);
    }
}



