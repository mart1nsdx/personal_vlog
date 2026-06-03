// ── DATA ────────────────────────────────────────────────────────────────────

const WHAT_I_DO = [
  {
    icon: "⚡",
    title: "Software Development",
    text: "I like building things that actually work. I move across the stack as needed and focus on shipping, not on making it perfect before anyone sees it.",
    tags: ["Full-Stack", "Product", "Shipping"],
  },
  {
    icon: "🤖",
    title: "AI & Automation",
    text: "I use AI and automation to get things done faster and smarter. I'm genuinely curious about what becomes possible when you put these tools in the right context.",
    tags: ["AI", "LLMs", "Automation", "Tooling"],
  },
  {
    icon: "🚀",
    title: "Entrepreneurship",
    text: "I care about finding real problems and building something around them. The early stage is where I feel most alive. Messy, uncertain, exciting.",
    tags: ["Startups", "Founder", "Strategy"],
  },
  {
    icon: "🌐",
    title: "Startup & Business Club · Uniandes",
    text: "I'm part of the Startup and Business Club at Universidad de los Andes. It's where I connect with other people building things and stay close to the entrepreneurship scene in LATAM.",
    tags: ["Uniandes", "Startups", "Community", "Entrepreneurship"],
  },
  {
    icon: "🛸",
    title: "Student Leader · Spacetech AESS IEEE Uniandes",
    text: "I lead the AESS chapter at Universidad de los Andes, the IEEE branch for Colombia. We get students together around aerospace engineering, orbital mechanics and rocket science. It's one of those groups where people actually geek out.",
    tags: ["IEEE", "AESS", "Spacetech", "Aerospace", "Uniandes", "Leadership"],
  },
];

const PROJECTS = [
  {
    num: "01",
    name: "Startup and Business Club Uniandes",
    role: "Product Developer",
    date: "Mar 2026 – Present · 4 mos",
    desc: "Building software and product integration with AI for our Startup community.",
    link: "https://www.executetheidea.com/",
    linkLabel: "executetheidea.com",
  },
  {
    num: "02",
    name: "Preicfes Orión · Self-employed",
    role: "Co-Founder",
    date: "Jun 2024 – Present · 2 yrs 1 mo · Colombia, Remote",
    desc: "Social & cultural impact of education // 3x Best National Score.",
  },
  {
    num: "03",
    name: "SpaceTech AESS UniAndes",
    role: "Team Lead",
    date: "Aug 2023 – Present · 2 yrs 11 mos",
    desc: "Teamwork in orbital mechanics, simulation, and AI/ML integration for rocketry.",
  },
];

const FUN_PROJECTS = [
  {
    emoji: "",
    name: "Hovercraft Prototype",
    desc: "Built a hovercraft from scratch. It hovers, it moves fast and detects obstacles in front of it using thermal and sound sensors.",
    image: "hovercraft.jpeg",
    video: "hovercraft_clean.mp4",
    tech: [],
  },
  {
    emoji: "",
    name: "Rocket Platform Simulation",
    desc: "Spacetech Uniandes Rocketry Research Group — simulation platform for rocket flight dynamics and trajectory modeling. An AI-powered bot is in the works to support mission decision-making and help group members grasp the fundamentals of orbital mechanics.",
    image: null,
    video: null,
    tech: [],
    images: ["rocket_sim_1.png", "rocket_sim_2.png", "rocket_sim_3.png"],
    github: "https://github.com/mart1nsdx/spacetech.git",
  },
  {
    emoji: "",
    name: "Space Collection",
    desc: "Models at scale are nice and cute — they represent the hardest and most exciting challenges for humankind. Collection in process.",
    image: "space_collection.jpeg",
    video: null,
    tech: [],
    members: ["NASA's Perseverance Mars Rover", "B2-Spirit \"Ghost\" 1:200", "SpaceX Starship"],
  },
];

const FUTURE_GOALS = [
  {
    icon: "🌎",
    title: "Developing ideas that drive real growth for LATAM",
    desc: "I want to build things that actually matter for Latin America. Hard tech, IoT, the stuff that's been missing. Not imported solutions adapted for the region but things built here from scratch, for here first.",
    horizon: "Long-term vision",
  },
];

const CONTACT_LINKS = [
  {
    icon: "💼",
    label: "LinkedIn",
    sub: "",
    href: "https://www.linkedin.com/in/martin-ardila-5bb02436b",
  },
];

// ── RENDER ───────────────────────────────────────────────────────────────────

function renderWhatIDo() {
  const grid = document.getElementById("what-i-do-cards");
  if (!grid) return;
  grid.innerHTML = WHAT_I_DO.map(
    ({ icon, title, text, tags }) => `
    <div class="card">
      <div class="card-icon">${icon}</div>
      <div class="card-title">${title}</div>
      <div class="card-text">${text}</div>
      <div class="card-tags">${tags.map(t => `<span class="tag">${t}</span>`).join("")}</div>
    </div>`
  ).join("");
}

function renderProjects() {
  const list = document.getElementById("projects-list");
  if (!list) return;
  list.innerHTML = PROJECTS.map(
    ({ num, name, role, date, desc, link, linkLabel }) => `
    <div class="project-item">
      <div class="project-num">${num}</div>
      <div class="project-content">
        <div class="project-name">${role} · <span style="color:var(--muted);font-weight:400">${name}</span></div>
        <div class="project-desc" style="margin-bottom:6px">${desc}</div>
        <div style="font-family:var(--mono);font-size:.68rem;color:var(--muted);letter-spacing:.06em">${date}</div>
      </div>
      ${link ? `<div class="project-meta"><a href="${link}" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:.7rem;letter-spacing:.08em;border:1px solid var(--border);color:var(--muted);padding:8px 16px;white-space:nowrap;transition:border-color .15s,color .15s" onmouseover="this.style.borderColor='#fff';this.style.color='#fff'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">${linkLabel} →</a></div>` : ''}
    </div>`
  ).join("");
}

function renderFunProjects() {
  const grid = document.getElementById("fun-projects-grid");
  if (!grid) return;
  grid.innerHTML = FUN_PROJECTS.map(
    ({ emoji, name, desc, image, video, tech, members, images, github }) => `
    <div class="fun-card">
      <div class="fun-body" style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px">
        <div>
          <div class="fun-name">${emoji} ${name}</div>
          <div class="fun-desc">${desc}</div>
          <div class="card-tags">${tech.map(t => `<span class="tag">${t}</span>`).join("")}</div>
        </div>
        ${github ? `<a href="${github}" target="_blank" rel="noopener" style="flex-shrink:0;display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;border:1px solid var(--border);color:var(--muted);padding:8px 16px;white-space:nowrap;transition:border-color .15s,color .15s" onmouseover="this.style.borderColor='#fff';this.style.color='#fff'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">⌥ GitHub Project →</a>` : ''}
      </div>
      ${images && images.length ? `
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;padding:0 28px 28px">
        ${images.map(src => `
        <div style="overflow:hidden;border:1px solid var(--border);background:var(--bg);aspect-ratio:4/3">
          <img src="${src}" alt="${name}" style="width:100%;height:100%;object-fit:cover;display:block;transition:transform .4s" onmouseover="this.style.transform='scale(1.04)'" onmouseout="this.style.transform='scale(1)'" />
        </div>`).join('')}
      </div>` : video ? `
      <div class="fun-media">
        <div class="fun-media-box"><img src="${image}" alt="${name}" class="fun-img" /></div>
        <div class="fun-media-box"><video controls class="fun-video"><source src="${video}" /></video></div>
      </div>` : `
      <div style="display:flex;gap:0;padding:0 28px 28px;align-items:flex-start">
        <div style="flex:0 0 55%;min-height:380px;border:1px solid var(--border);overflow:hidden;background:var(--bg)">
          <img src="${image}" alt="${name}" style="width:100%;height:100%;object-fit:cover;object-position:center;display:block;" />
        </div>
        ${members && members.length ? `
        <div style="flex:1;padding:0 0 0 28px">
          <div style="font-family:var(--mono);font-size:.68rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px">Family members</div>
          <ul style="list-style:none;display:flex;flex-direction:column;gap:10px">
            ${members.map(m => `<li style="font-size:.88rem;color:var(--text);display:flex;align-items:center;gap:10px"><span style="color:var(--accent);font-family:var(--mono)">—</span>${m}</li>`).join('')}
          </ul>
        </div>` : ''}
      </div>`}
    </div>`
  ).join("");
}

function renderFuture() {
  const grid = document.getElementById("future-grid");
  if (!grid) return;
  grid.innerHTML = FUTURE_GOALS.map(
    ({ icon, title, desc, horizon }) => `
    <div class="future-item">
      <div class="future-icon">${icon}</div>
      <div class="future-text">
        <div class="title">${title}</div>
        <div class="desc">${desc}</div>
        <div class="future-horizon">⟶ ${horizon}</div>
      </div>
    </div>`
  ).join("");
}

function renderContact() {
  const wrap = document.getElementById("contact-links");
  if (!wrap) return;
  wrap.innerHTML = CONTACT_LINKS.map(
    ({ icon, label, sub, href }) => `
    <a href="${href}" class="contact-link" target="_blank" rel="noopener">
      <span class="icon">${icon}</span>
      <span class="link-label">${label}${sub ? `<br/><small style="color:var(--muted);font-size:.75rem;">${sub}</small>` : ''}</span>
      <span class="link-arrow">→</span>
    </a>`
  ).join("");
}

// ── ANIMATIONS ───────────────────────────────────────────────────────────────

function initScrollReveal() {
  const items = document.querySelectorAll(".card, .project-item, .future-item, .contact-link");
  items.forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(24px)";
    el.style.transition = "opacity .5s ease, transform .5s ease";
  });

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = "1";
          e.target.style.transform = "translateY(0)";
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  items.forEach((el, i) => {
    setTimeout(() => observer.observe(el), i * 60);
  });
}

function initNavScroll() {
  const nav = document.querySelector("nav");
  window.addEventListener("scroll", () => {
    nav.style.background = window.scrollY > 40
      ? "rgba(2,4,9,0.92)"
      : "rgba(2,4,9,0.7)";
  });
}

function initOrbParallax() {
  const orb1 = document.querySelector(".orb-1");
  const orb2 = document.querySelector(".orb-2");
  window.addEventListener("mousemove", e => {
    const x = (e.clientX / window.innerWidth - 0.5) * 30;
    const y = (e.clientY / window.innerHeight - 0.5) * 30;
    if (orb1) orb1.style.transform = `translate(${x}px, ${y}px)`;
    if (orb2) orb2.style.transform = `translate(${-x}px, ${-y}px)`;
  });
}

function initTypewriter() {
  const subtitle = document.querySelector(".hero-subtitle");
  if (!subtitle) return;
  const text = subtitle.textContent;
  subtitle.textContent = "";
  let i = 0;
  const type = () => {
    if (i < text.length) {
      subtitle.textContent += text[i++];
      setTimeout(type, 40);
    }
  };
  setTimeout(type, 600);
}

// ── INIT ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  renderWhatIDo();
  renderProjects();
  renderFunProjects();
  renderFuture();
  renderContact();

  initScrollReveal();
  initNavScroll();
  initOrbParallax();
  initTypewriter();
});
