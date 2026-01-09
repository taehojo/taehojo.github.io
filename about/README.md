---
layout: default
title: About Taeho Jo
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

<style>
.profile-section {
  display: flex;
  align-items: flex-start;
  gap: 2rem;
  margin-bottom: 2rem;
  flex-wrap: wrap;
}
.profile-image {
  flex-shrink: 0;
}
.profile-image img {
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.profile-intro {
  flex: 1;
  min-width: 280px;
}
.profile-intro h1 {
  margin-top: 0;
  margin-bottom: 0.5rem;
}
.title-badge {
  display: inline-block;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.85rem;
  margin-bottom: 1rem;
}
.contact-links {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
  flex-wrap: wrap;
}
.contact-links a {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.5rem 1rem;
  background: #f5f5f5;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  transition: all 0.2s;
}
.contact-links a:hover {
  background: #e8e8e8;
  transform: translateY(-2px);
}
.section-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 2.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #eee;
}
.research-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}
.research-card {
  background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
  border: 1px solid #e9ecef;
  border-radius: 12px;
  padding: 1.2rem;
  transition: all 0.2s;
}
.research-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  transform: translateY(-2px);
}
.research-card h4 {
  margin: 0 0 0.5rem 0;
  color: #495057;
}
.research-card p {
  margin: 0;
  font-size: 0.9rem;
  color: #6c757d;
}
.tool-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.8rem;
}
.tool-tag {
  background: #e9ecef;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-size: 0.75rem;
  color: #495057;
}
.book-list {
  list-style: none;
  padding: 0;
}
.book-list li {
  padding: 0.8rem 0;
  border-bottom: 1px solid #f0f0f0;
}
.book-list li:last-child {
  border-bottom: none;
}
.book-year {
  display: inline-block;
  background: #f0f0f0;
  padding: 0.1rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  color: #666;
  margin-left: 0.5rem;
}
.book-new {
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
  color: white;
}
.book-links {
  margin-top: 0.3rem;
}
.book-links a {
  font-size: 0.85rem;
  margin-right: 0.8rem;
  color: #667eea;
}
.paper-list {
  list-style: none;
  padding: 0;
}
.paper-list li {
  padding: 0.6rem 0;
  padding-left: 1.5rem;
  position: relative;
  border-bottom: 1px solid #f5f5f5;
}
.paper-list li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 1rem;
  width: 6px;
  height: 6px;
  background: #667eea;
  border-radius: 50%;
}
.paper-list li:last-child {
  border-bottom: none;
}
</style>

<div class="profile-section">
  <div class="profile-image">
    <img src="https://avatars.githubusercontent.com/u/28382194?v=4" alt="Taeho Jo" width="180">
  </div>
  <div class="profile-intro">
    <h1>Taeho Jo, Ph.D.</h1>
    <h1 style="font-size: 1.5rem; color: #666; margin-top: 0;">조태호</h1>
    <span class="title-badge">Assistant Professor, Indiana University School of Medicine</span>
    <p>AI/딥러닝을 활용하여 <strong>알츠하이머병의 조기 진단</strong>을 연구합니다.<br>연구와 함께 IT 서적을 집필하며 지식을 나누고 있습니다.</p>
    <div class="contact-links">
      <a href="https://www.jolab.ai" target="_blank">🔬 JoLab.ai</a>
      <a href="https://github.com/taehojo" target="_blank">💻 GitHub</a>
    </div>
  </div>
</div>

---

<div class="section-title">
  <span style="font-size: 1.5rem;">🧬</span>
  <h2 style="margin: 0;">Research Areas</h2>
</div>

<div class="research-grid">
  <div class="research-card">
    <h4>Genomics & AI</h4>
    <p>대규모 유전체 데이터와 딥러닝을 결합하여 알츠하이머 관련 유전 변이를 식별합니다.</p>
    <div class="tool-tags">
      <span class="tool-tag">TrUE-Net</span>
      <span class="tool-tag">Deep-Block</span>
      <span class="tool-tag">SWAT-CNN</span>
    </div>
  </div>
  <div class="research-card">
    <h4>Neuroimaging & AI</h4>
    <p>PET/MRI 영상과 3D CNN을 활용하여 뇌의 미세한 변화를 조기에 감지합니다.</p>
    <div class="tool-tags">
      <span class="tool-tag">3D CNN</span>
      <span class="tool-tag">PET</span>
      <span class="tool-tag">MRI</span>
    </div>
  </div>
  <div class="research-card">
    <h4>Metabolomics & AI</h4>
    <p>혈액 기반 바이오마커를 분석하여 질병 진행을 예측하고 조기 진단에 활용합니다.</p>
    <div class="tool-tags">
      <span class="tool-tag">c-SWAT</span>
      <span class="tool-tag">Biomarker</span>
    </div>
  </div>
  <div class="research-card">
    <h4>Precision Medicine</h4>
    <p>유전체, 신경영상, 대사체학을 AI와 통합하여 질병 진행을 정밀하게 예측하고, 개인 맞춤형 조기 진단을 개발합니다.</p>
    <div class="tool-tags">
      <span class="tool-tag">BIBio</span>
      <span class="tool-tag">AAIC25</span>
      <span class="tool-tag">AAIC23</span>
      <span class="tool-tag">TRCI</span>
    </div>
  </div>
</div>

<div class="section-title">
  <span style="font-size: 1.5rem;">📚</span>
  <h2 style="margin: 0;">Publications - Books</h2>
</div>

**IT Books**

<ul class="book-list">
  <li>
    <strong>혼자 공부하는 바이브 코딩 with 클로드 코드</strong> <span class="book-year book-new">2025 NEW</span>
    <div class="book-links">
      <a href="https://github.com/taehojo/vibecoding" target="_blank">Code</a>
      <a href="http://www.yes24.com/Product/Search?domain=ALL&query=혼자공부하는바이브코딩" target="_blank">Book</a>
    </div>
  </li>
  <li>
    <strong>모두의 딥러닝</strong> 개정4판 <span class="book-year book-new">2025 NEW</span>
    <div class="book-links">
      <a href="https://github.com/taehojo/deeplearning" target="_blank">Code</a>
      <a href="http://www.yes24.com/Product/Search?domain=ALL&query=모두의딥러닝" target="_blank">Book</a>
      <a href="https://www.youtube.com/@taehojo" target="_blank">YouTube</a>
    </div>
  </li>
  <li>
    <strong>모두의 딥러닝</strong> 개정3판 <span class="book-year">2022</span>
    <div class="book-links">
      <a href="https://github.com/taehojo/deeplearning" target="_blank">Code</a>
      <a href="http://www.yes24.com/Product/Goods/108553440" target="_blank">Book</a>
    </div>
  </li>
  <li>
    <strong>그림으로 배우는 인지과학</strong> <span class="book-year">2022</span>
    <div class="book-links">
      <a href="http://www.yes24.com/Product/Goods/108250950" target="_blank">Book</a>
    </div>
  </li>
  <li>
    <strong>쉽게 시작하는 캐글 데이터 분석</strong> <span class="book-year">2021</span>
    <div class="book-links">
      <a href="https://github.com/taehojo/getting_started_with_kaggle" target="_blank">Code</a>
      <a href="https://taehojo.github.io/book/kaggle-092322.pdf" target="_blank">Errata</a>
      <a href="http://www.yes24.com/Product/Goods/103526120" target="_blank">Book</a>
    </div>
  </li>
  <li>
    <strong>딥러닝 워크북</strong> <span class="book-year">2018</span>
    <div class="book-links">
      <a href="https://github.com/taehojo/deeplearning-workshop" target="_blank">Code</a>
      <a href="http://www.yes24.com/Product/Goods/59789570" target="_blank">Book</a>
    </div>
  </li>
  <li>
    <strong>모두의 딥러닝</strong> 개정2판 / 1판 <span class="book-year">2019 / 2017</span>
    <div class="book-links">
      <a href="https://github.com/taehojo/deeplearning-for-everyone-2nd" target="_blank">Code (2nd)</a>
      <a href="https://github.com/taehojo/deeplearning-for-everyone-1st" target="_blank">Code (1st)</a>
    </div>
  </li>
</ul>

**Essay**

<ul class="book-list">
  <li>
    🏆 <strong>당신의 이유는 무엇입니까</strong> - 제7회 브런치북 대상 수상작 <span class="book-year">2020</span>
    <div class="book-links">
      <a href="http://www.yes24.com/Product/Goods/90981164" target="_blank">Book</a>
      <a href="https://brunch.co.kr/brunchbook/not-this-world" target="_blank">Brunch</a>
      <a href="https://www.youtube.com/watch?v=szHI91_ZbBU" target="_blank">Radio</a>
    </div>
  </li>
</ul>

<div class="section-title">
  <span style="font-size: 1.5rem;">🔬</span>
  <h2 style="margin: 0;">Publications - Research</h2>
</div>

**유전체 & AI**

<ul class="paper-list">
  <li><strong>TrUE-Net</strong> - 트랜스포머 기반 알츠하이머 유전 변이 식별 <span class="book-year book-new">2025</span><br><span style="color: #6c757d; font-size: 0.85rem;">Briefings in Bioinformatics | 72.9% 정확도</span> <a href="https://github.com/taehojo/TrUE-Net" target="_blank">[Code]</a></li>
  <li><strong>Deep-Block</strong> - 딥러닝 기반 LD 블록 유전 변이 분석 <span class="book-year book-new">2025</span><br><span style="color: #6c757d; font-size: 0.85rem;">Alzheimer's & Dementia TRCI | 30,218개 LD 블록 식별</span> <a href="https://github.com/taehojo/Deep-Block" target="_blank">[Code]</a></li>
  <li><strong>SWAT-CNN</strong> - 슬라이딩 윈도우 기반 유전 변이 식별 <span class="book-year">2022</span><br><span style="color: #6c757d; font-size: 0.85rem;">Briefings in Bioinformatics | AUC 0.82</span> <a href="https://github.com/taehojo/SWAT" target="_blank">[Code]</a></li>
</ul>

**신경영상 & AI**

<ul class="paper-list">
  <li><strong>Tau PET 분류</strong> - 3D CNN 기반 타우 단백질 PET 영상 분류 <span class="book-year">2020</span><br><span style="color: #6c757d; font-size: 0.85rem;">BMC Bioinformatics | 90.8% 정확도</span></li>
</ul>

**단백질체 & AI**

<ul class="paper-list">
  <li><strong>종단 단백질체 분석</strong> - 알츠하이머 진행 예측 모델 <span class="book-year book-new">2025</span><br><span style="color: #6c757d; font-size: 0.85rem;">Alzheimer's & Dementia | AUC 0.848</span></li>
</ul>

**대사체 & AI**

<ul class="paper-list">
  <li><strong>c-SWAT</strong> - 대사체 기반 알츠하이머 바이오마커 식별 <span class="book-year">2023</span><br><span style="color: #6c757d; font-size: 0.85rem;">eBioMedicine (Lancet) | 80.8% 정확도</span> <a href="https://github.com/taehojo/c-SWAT" target="_blank">[Code]</a></li>
</ul>

<p style="text-align: center; margin-top: 2rem; color: #999;">
  <a href="https://www.jolab.ai" target="_blank" style="color: #667eea;">More research at JoLab.ai →</a>
</p>
