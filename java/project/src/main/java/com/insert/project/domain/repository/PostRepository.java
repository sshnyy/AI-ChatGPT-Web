package com.insert.project.domain.repository;

import com.insert.project.domain.entity.BoardEntity;
import com.insert.project.dto.BoardDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import javax.persistence.EntityManager;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class PostRepository {
    private final EntityManager em;

    // 마이페이지에서 챗로그만 빼고 불러옴
    public List<BoardEntity> findByNameExcludingTitle(Long memberId, String titleToExclude) {
        return em.createQuery("SELECT b FROM BoardEntity b WHERE b.member.id = :memberId AND b.boardTitle != :titleToExclude", BoardEntity.class)
                .setParameter("memberId", memberId)
                .setParameter("titleToExclude", titleToExclude)
                .getResultList();
    }
    // 나의 채팅 목록만 불러옴
    public List<BoardEntity> findMyChatLogPosts(Long memberId) {
        return em.createQuery("SELECT b FROM BoardEntity b WHERE b.member.id = :memberId AND b.boardTitle = 'chat_log'", BoardEntity.class)
                .setParameter("memberId", memberId)
                .getResultList();
    }

    // 게시판 불러오기(채팅로그 빼고)
    public List<BoardEntity> findAllPostsExcludingTitle(String titleToExclude) {
        return em.createQuery("SELECT b FROM BoardEntity b WHERE b.boardTitle != :titleToExclude", BoardEntity.class)
                .setParameter("titleToExclude", titleToExclude)
                .getResultList();
    }


//    public List<BoardEntity> findByNameExcludingChat(Long memberId, String titleToExclude) {
//        return em.createQuery("SELECT b FROM BoardEntity b WHERE b.member.id = :memberId AND b.boardTitle = :titleToExclude", BoardEntity.class)
//                .setParameter("memberId", memberId)
//                .setParameter("titleToExclude", titleToExclude)
//                .getResultList();
//    }




}
