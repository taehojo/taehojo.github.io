---
layout: default
title: Home
nav_exclude: true
---

<div id="custom-blog-header">
  <div class="header-inner">
    <a href="/" class="site-name"><img src="/logo.png" alt="AI & Life Science"><span class="blog-badge">blog</span></a>
    <nav>
      <a href="/about/">About Taeho Jo</a>
      <a href="https://www.jolab.ai" target="_blank">JoLab.ai</a>
    </nav>
  </div>
</div>

{% for post in site.posts %}
# {{ post.title }}

{{ post.content }}

<div class="post-end">
<p class="post-date">{{ post.date | date: "%Y년 %-m월" }}</p>
</div>

{% unless forloop.last %}
<div class="post-divider">
<span>다음 글</span>
</div>
{% endunless %}

{% endfor %}

<div class="author-box">
<img src="/about/images/taehojo.png" alt="조태호" class="author-photo">
<div class="author-info">
<p class="author-name">조태호 (Taeho Jo, PhD)</p>
<p class="author-title">인디애나대학교 의과대학 영상의학과 교수</p>
<p class="author-bio"><a href="https://www.yes24.com/product/goods/145612410" target="_blank">모두의 딥러닝</a>, <a href="https://www.yes24.com/product/goods/167573138" target="_blank">혼자공부하는 바이브코딩</a>, 당신의 이유는 무엇입니까 저자.</p>
<p class="author-links">
<a href="https://www.jolab.ai" target="_blank">JoLab.ai</a>
<a href="https://github.com/taehojo" target="_blank">GitHub</a>
<a href="https://www.youtube.com/@taehojo" target="_blank">YouTube</a>
<a href="mailto:taehjo@gmail.com">Email</a>
</p>
</div>
</div>

<p class="write-link"><a href="/guide/">write</a></p>

<p class="copyright">© 2025 Taeho Jo. All rights reserved.</p>
