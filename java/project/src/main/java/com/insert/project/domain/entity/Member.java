package com.insert.project.domain.entity;


import lombok.Getter;
import lombok.Setter;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@Setter
public class Member {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "member_id")
    private Long id;

    private String loginId;
    private String name;
    private String password;
    private int age;
    private String gender;
    private String address;

    @OneToMany(mappedBy = "member", cascade = CascadeType.ALL)
    private List<BoardEntity> boardEntityList = new ArrayList<>();
}
