package com.insert.project.service;

import com.insert.project.domain.entity.Member;
import com.insert.project.domain.repository.MemberRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.util.List;

@Service
public class MemberService {
    @Autowired
    private  MemberRepository memberRepository;


    public List<Member> findMembers() {
        return memberRepository.findAll();
    }
    public Member findOne(Long Id) {
        return memberRepository.findOne(Id);
    }

    @Transactional
    public Member update(Member member, Member loginMember){
        Member findMember = memberRepository.findOne(loginMember.getId());
        findMember.setAge(member.getAge());
        findMember.setName(member.getName());
        findMember.setLoginId(member.getLoginId());
        findMember.setPassword(member.getPassword());
        findMember.setAddress(member.getAddress());
        findMember.setGender(member.getGender());
        return findMember;
    }

}
