package com.insert.project.domain.repository;

import com.insert.project.domain.entity.BoardEntity;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface BoardRepository extends JpaRepository<BoardEntity, Long> {

    @Modifying
    @Query(value = "update BoardEntity b set b.boardHits=b.boardHits+1 where b.id=:id")
    void updateHits(@Param("id") Long id);

    @Query("SELECT b FROM BoardEntity b WHERE b.boardTitle != :excludeTitle")
    Page<BoardEntity> findAllExcludeChatLog(@Param("excludeTitle") String excludeTitle, Pageable pageable);
}
