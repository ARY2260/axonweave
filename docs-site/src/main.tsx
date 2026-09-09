import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Helmet, HelmetProvider } from 'react-helmet-async';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { Menu, Moon, Sun, Copy, Check, Search, X } from 'lucide-react';
import './style.css';

import indexMd from '../content/index.md?raw';
import gettingStartedMd from '../content/getting-started.md?raw';
import architectureMd from '../content/architecture.md?raw';
import backendsMd from '../content/backends.md?raw';
import biologyMd from '../content/biology.md?raw';
import interoperabilityMd from '../content/interoperability.md?raw';
import distributionMd from '../content/distribution.md?raw';
import scientificMd from '../content/scientific-reference.md?raw';
import releaseMd from '../content/release-engineering.md?raw';
import contributingMd from '../content/contributing.md?raw';
import apiMd from '../content/api-reference.md?raw';
import troubleshootingMd from '../content/troubleshooting.md?raw';
import configurationMd from '../content/configuration.md?raw';
import privacyMd from '../content/privacy.md?raw';
import termsMd from '../content/terms.md?raw';
import examplesCustomPolicyMd from '../content/examples-custom-policy.md?raw';
import examplesPytorchCompositionMd from '../content/examples-pytorch-composition.md?raw';

type Page = { slug: string; label: string; source: string; section: string };
const pages: Page[] = [
 {slug:'index',label:'Overview',source:indexMd,section:'Start'},
 {slug:'getting-started',label:'Getting Started',source:gettingStartedMd,section:'Start'},
 {slug:'architecture',label:'Architecture',source:architectureMd,section:'Core'},
 {slug:'backends',label:'Backends & Devices',source:backendsMd,section:'Core'},
 {slug:'biology',label:'Biological Model',source:biologyMd,section:'Science'},
 {slug:'interoperability',label:'Interoperability',source:interoperabilityMd,section:'Science'},
 {slug:'distribution',label:'Substrate Distribution',source:distributionMd,section:'Operations'},
 {slug:'scientific-reference',label:'Scientific Reference',source:scientificMd,section:'Science'},
 {slug:'examples-pytorch-composition',label:'Example: PyTorch Composition',source:examplesPytorchCompositionMd,section:'Examples'},
 {slug:'examples-custom-policy',label:'Example: Custom Signal Policy',source:examplesCustomPolicyMd,section:'Examples'},
 {slug:'release-engineering',label:'Release Engineering',source:releaseMd,section:'Operations'},
 {slug:'api-reference',label:'API Reference',source:apiMd,section:'Reference'},
 {slug:'troubleshooting',label:'Troubleshooting',source:troubleshootingMd,section:'Reference'},
 {slug:'configuration',label:'Configuration',source:configurationMd,section:'Reference'},
 {slug:'contributing',label:'Contributing',source:contributingMd,section:'Project'},
 {slug:'privacy',label:'Privacy Policy',source:privacyMd,section:'Project'},
 {slug:'terms',label:'Terms & Conditions',source:termsMd,section:'Project'},
];

marked.setOptions({gfm:true, breaks:false});
const normalize = (md:string) => md.replace(/^:::DOC-NOTE\n([\s\S]*?)\n:::/gm, '> **Note**\n> $1').replace(/^:::DOC-WARN\n([\s\S]*?)\n:::/gm, '> **Constraint**\n> $1');
const render = (md:string) => DOMPurify.sanitize(marked.parse(normalize(md)) as string, {ADD_ATTR:['target','rel']});
const BASE = import.meta.env.BASE_URL;
const BASE_PREFIX = BASE.endsWith('/') ? BASE.slice(0,-1) : BASE;
const hrefFor = (slug:string) => `${BASE}${slug==='index'?'':slug}`;
const pathSlug = () => location.pathname.replace(BASE_PREFIX,'').replace(/^\//,'').replace(/\/$/,'') || 'index';

function CodeEnhancer(){
 const [copied,setCopied]=useState<number|null>(null);
 useEffect(()=>{
   const blocks=[...document.querySelectorAll('pre')];
   blocks.forEach((pre,i)=>{
     if(pre.querySelector('button')) return;
     const button=document.createElement('button'); button.className='copy-button'; button.setAttribute('aria-label','Copy code'); button.innerHTML='<span class="copy-label">Copy</span>';
     button.onclick=async()=>{await navigator.clipboard.writeText(pre.querySelector('code')?.textContent||'');setCopied(i);setTimeout(()=>setCopied(null),1400)};
     pre.appendChild(button);
   });
   return ()=>{};
 },[]);
 return <span className="copy-status" aria-live="polite">{copied!==null?'Copied':''}</span>;
}

function App(){
 const [slug,setSlug]=useState(pathSlug());
 const [dark,setDark]=useState(localStorage.getItem('axonweave-theme')!=='light');
 const [mobile,setMobile]=useState(false);
 const [searchOpen,setSearchOpen]=useState(false);
 useEffect(()=>{const fn=()=>setSlug(pathSlug()); addEventListener('popstate',fn);return()=>removeEventListener('popstate',fn)},[]);
 useEffect(()=>{document.documentElement.dataset.theme=dark?'dark':'light';localStorage.setItem('axonweave-theme',dark?'dark':'light')},[dark]);
 const page=pages.find(p=>p.slug===slug);
 const html=useMemo(()=>page?render(page.source):'', [page]);
 useEffect(()=>{document.title=page?`${page.label} · AxonWeave`: 'Page not found · AxonWeave'; const domain=import.meta.env.VITE_ANALYTICS_DOMAIN as string|undefined; if(domain && !document.querySelector('script[data-axonweave-analytics]')){const script=document.createElement('script');script.defer=true;script.dataset.domain=domain;script.dataset.axonweaveAnalytics='true';script.src=`https://plausible.io/js/script.js`;document.head.appendChild(script)}},[page]);
 const sections=[...new Set(pages.map(p=>p.section))];
 const navigate=(s:string)=>{history.pushState({},'',hrefFor(s));setSlug(s);setMobile(false);scrollTo(0,0)};
 return <><Helmet><meta name="description" content={page?`AxonWeave ${page.label} documentation`: 'AxonWeave documentation'}/><meta property="og:title" content={page?`${page.label} · AxonWeave`:'AxonWeave Documentation'}/><meta property="og:description" content={page?`AxonWeave ${page.label} documentation`:'AxonWeave documentation'}/><meta name="twitter:card" content="summary"/></Helmet><div className="app">
   <header className="topbar">
     <button className="icon-button mobile-menu" onClick={()=>setMobile(!mobile)} aria-label="Open navigation"><Menu size={20}/></button>
     <a className="brand" href="/" onClick={e=>{e.preventDefault();navigate('index')}}><img src={`${BASE}logo.svg`} alt="AxonWeave"/><span>AxonWeave</span></a>
     <nav className="topnav"><a href={hrefFor('getting-started')} onClick={e=>{e.preventDefault();navigate('getting-started')}}>Docs</a><a href="https://github.com/dhakalnirajan/axonweave">GitHub</a><div className="learn-menu"><button className="learn-trigger">Learn <span>▾</span></button><div className="learn-dropdown"><a href={hrefFor('biology')} onClick={e=>{e.preventDefault();navigate('biology')}}>Biological Model</a><a href={hrefFor('backends')} onClick={e=>{e.preventDefault();navigate('backends')}}>Backends &amp; Devices</a><a href={hrefFor('interoperability')} onClick={e=>{e.preventDefault();navigate('interoperability')}}>Interoperability</a><a href={hrefFor('examples-pytorch-composition')} onClick={e=>{e.preventDefault();navigate('examples-pytorch-composition')}}>PyTorch Example</a></div></div><a href={hrefFor('scientific-reference')} onClick={e=>{e.preventDefault();navigate('scientific-reference')}}>Scientific Reference</a></nav>
     <div className="top-actions"><button className="icon-button" onClick={()=>setSearchOpen(true)} aria-label="Search documentation"><Search size={19}/></button><button className="icon-button" onClick={()=>setDark(!dark)} aria-label="Toggle theme">{dark?<Sun size={19}/>:<Moon size={19}/>}</button></div>
   </header>
   <div className="shell">
    <aside className={`sidebar ${mobile?'open':''}`}>
      <div className="sidebar-header">Documentation <button className="icon-button close-mobile" onClick={()=>setMobile(false)}><X size={18}/></button></div>
      {sections.map(section=><div className="nav-section" key={section}><div className="nav-label">{section}</div>{pages.filter(p=>p.section===section).map(p=><a key={p.slug} className={slug===p.slug?'active':''} href={hrefFor(p.slug)} onClick={e=>{e.preventDefault();navigate(p.slug)}}>{p.label}</a>)}</div>)}
    </aside>
    <main id="main" className="content">
      {page?<><div className="breadcrumbs">Docs <span>/</span> {page.label}</div><article dangerouslySetInnerHTML={{__html:html}}/>{slug==='index'&&<div className="hero-cta"><button className="primary" onClick={()=>navigate('getting-started')}>Install AxonWeave</button></div>}<CodeEnhancer/><div className="page-nav"><span>AxonWeave Documentation</span><button className="text-button" onClick={()=>navigate('getting-started')}>Continue →</button></div></>:<NotFound navigate={navigate}/>} 
    </main>
    {page&&<aside className="toc"><div className="toc-title">On this page</div><Toc/></aside>}
   </div>
   <footer><span>AxonWeave · Apache-2.0 software</span><span><a href={hrefFor('privacy')} onClick={e=>{e.preventDefault();navigate('privacy')}}>Privacy</a> · <a href={hrefFor('terms')} onClick={e=>{e.preventDefault();navigate('terms')}}>Terms</a></span></footer>
   <CookieConsent/>
   {searchOpen&&<SearchDialog pages={pages} onClose={()=>setSearchOpen(false)} onGo={navigate}/>}
  </div></>
}

function Toc(){const [items,setItems]=useState<{id:string,text:string,level:number}[]>([]);useEffect(()=>{const hs=[...document.querySelectorAll('article h2,article h3')];const out=hs.map((h,i)=>{const id=`section-${i}-${(h.textContent||'').toLowerCase().replace(/[^a-z0-9]+/g,'-')}`;h.id=id;return{id,text:h.textContent||'',level:h.tagName==='H2'?2:3}});setItems(out)},[]);return <nav>{items.map(x=><a className={x.level===3?'sub':''} href={`#${x.id}`} key={x.id}>{x.text}</a>)}</nav>}
function NotFound({navigate}:{navigate:(s:string)=>void}){return <div className="not-found"><p className="eyebrow">404</p><h1>Page not found</h1><p>The requested documentation page does not exist.</p><button className="primary" onClick={()=>navigate('index')}>Return to documentation</button></div>}
function SearchDialog({pages,onClose,onGo}:{pages:Page[],onClose:()=>void,onGo:(s:string)=>void}){const [q,setQ]=useState('');const results=pages.filter(p=>(p.label+' '+p.source).toLowerCase().includes(q.toLowerCase())).slice(0,8);return <div className="overlay" onMouseDown={onClose}><div className="search-dialog" onMouseDown={e=>e.stopPropagation()}><div className="search-head"><Search size={18}/><input autoFocus value={q} onChange={e=>setQ(e.target.value)} placeholder="Search documentation"/><button className="icon-button" onClick={onClose}><X size={18}/></button></div>{results.map(r=><button className="search-result" key={r.slug} onClick={()=>{onGo(r.slug);onClose()}}><strong>{r.label}</strong><span>{r.section}</span></button>)}</div></div>}
function CookieConsent(){const [show,setShow]=useState(localStorage.getItem('axonweave-cookie')!=='accepted');if(!show)return null;return <div className="cookie"><div><strong>Privacy choices</strong><p>This documentation does not require analytics cookies. Optional analytics, if enabled by a deployment, should be disclosed in the Privacy Policy.</p></div><button className="primary" onClick={()=>{localStorage.setItem('axonweave-cookie','accepted');setShow(false)}}>Accept</button></div>}

createRoot(document.getElementById('root')!).render(<HelmetProvider><App/></HelmetProvider>);
