---
layout: default
title: 글쓰기 가이드
nav_exclude: true
permalink: /guide/
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

# 블로그 글쓰기 가이드

## 새 글 작성 방법

`_posts` 폴더에 다음 형식으로 파일을 만듭니다:

```
YYYY-MM-DD-제목.md
```

예: `2026-01-08-new-post.md`

## 글 형식

파일 상단에 아래 내용을 추가합니다:

```markdown
---
layout: post
title: "글 제목"
date: 2026-01-08
description: "글에 대한 간단한 설명"
---

여기에 본문을 작성합니다...
```

## 이미지 추가

이미지는 `/alphafold/images/` 폴더에 저장하고 다음과 같이 사용합니다:

```markdown
![이미지 설명](/alphafold/images/파일명.png)
```

## 코드 삽입

코드 블록은 다음과 같이 작성합니다:

~~~markdown
```python
def hello():
    print("Hello, World!")
```
~~~

## 인용문

```markdown
> 인용하고 싶은 내용을 여기에 작성합니다.
```

---

글을 저장하고 GitHub에 푸시하면 자동으로 블로그에 반영됩니다.
