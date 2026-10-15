import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowRight, GitBranch, SlidersHorizontal, Sparkles, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import './index.css';

const fields = [
 ['cgpa','CGPA','Out of 10 · 8.5+ is a strong academic profile',0,10,.1],
 ['coding_questions_solved','Coding questions solved','Unique problems · 150 is a demo practice benchmark',0,5000,1],
 ['internships','Internships','Completed internships · one already counts strongly',0,10,1],
 ['leadership_roles','Leadership roles','Club, team or event roles · zero is common',0,10,1],
 ['project_complexity','Project complexity','1: tutorial · 2: basic · 3: independent · 4: advanced · 5: research',1,5,1],
 ['attendance_percent','Attendance (%)','Attendance percentage, from 0 to 100',0,100,1],
 ['academic_gaps','Academic gap years','Completed gap years, not backlog count',0,10,1],
];
const labels = Object.fromEntries(fields.map(f=>[f[0],f[1]]));
const examples = {
 'Strong':{cgpa:9,coding_questions_solved:200,internships:1,leadership_roles:0,project_complexity:3,attendance_percent:90,academic_gaps:0},
 'Developing':{cgpa:8,coding_questions_solved:100,internships:0,leadership_roles:1,project_complexity:3,attendance_percent:80,academic_gaps:0},
 'Starting out':{cgpa:6,coding_questions_solved:50,internships:0,leadership_roles:0,project_complexity:2,attendance_percent:70,academic_gaps:0}
};
const titles = {'1':'Strong readiness','2':'Developing readiness','3':'Foundation building'};
function Bars({values, categories=false}) { return <div className="space-y-4">{Object.entries(values).sort((a,b)=>categories?Number(a[0])-Number(b[0]):b[1]-a[1]).map(([k,v])=><div key={k}><div className="mb-2 flex justify-between text-sm"><span>{categories?`Category ${k}`:labels[k]}</span><span className="tabular-nums text-muted-foreground">{(v*100).toFixed(1)}%</span></div><Progress value={v*100} className="h-1.5"/></div>)}</div> }
function App(){
 const [form,setForm]=useState(examples.Strong),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState(''),[snapshot,setSnapshot]=useState(null);
 const stale=result&&JSON.stringify(form)!==JSON.stringify(snapshot);
 async function submit(e){ e.preventDefault(); setBusy(true);setError('');setResult(null);const submitted={...form};try{const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(Object.entries(submitted).map(([k,v])=>[k,Number(v)])))});const data=await response.json();if(!response.ok)throw new Error(data.error||'Prediction failed.');setResult(data);setSnapshot(submitted);}catch(err){setError(`Could not process this profile. ${err.message}`);}finally{setBusy(false);} }
 return <div>
  <main className="mx-auto max-w-7xl px-5 py-9 md:px-6">
   <div className="mb-8 flex flex-wrap items-end justify-between gap-4"><div><p className="mb-2 text-xs font-semibold uppercase tracking-[.18em] text-primary">Student readiness / Explainable ML</p><h1 className="text-3xl font-semibold tracking-tight md:text-4xl">A category. And the reason behind it.</h1><p className="mt-3 text-sm text-muted-foreground">Enter a student profile to see what the decision tree predicts and how it gets there.</p></div><span className="flex items-center gap-2 text-xs text-muted-foreground"><GitBranch size={15}/> Decision tree classifier</span></div>
   <div className="grid items-start gap-6 lg:grid-cols-[.9fr_1.1fr]">
    <Card className="shadow-none"><CardHeader className="border-b"><div className="flex items-center justify-between"><CardTitle className="flex items-center gap-2 text-lg"><SlidersHorizontal size={18}/> Student profile</CardTitle><span className="text-xs text-muted-foreground">01 / Input</span></div><CardDescription>Use your own numbers or try a sample profile.</CardDescription><div className="mt-2 flex flex-wrap gap-2">{Object.entries(examples).map(([name,value])=><Button key={name} type="button" variant="outline" size="sm" disabled={busy} onClick={()=>{setForm({...value});setError('');}}>{name}</Button>)}</div></CardHeader>
     <CardContent><form onSubmit={submit} className="space-y-5"><fieldset disabled={busy} className="grid gap-5 sm:grid-cols-2">{fields.map(([key,label,hint,min,max,step])=><div key={key} className={key==='project_complexity'?'sm:col-span-2':''}><Label htmlFor={key} className="mb-2 block">{label}</Label><Input id={key} type="number" min={min} max={max} step={step} required value={form[key]} onChange={e=>setForm({...form,[key]:e.target.value})} aria-describedby={`${key}-hint`} className="h-11 bg-white"/><p id={`${key}-hint`} className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{hint}</p></div>)}</fieldset><Button type="submit" disabled={busy} className="h-12 w-full">{busy?'Reading the decision tree…':'Predict category'}<ArrowRight size={17}/></Button>{error&&<p role="alert" className="text-sm text-destructive">{error}</p>}</form></CardContent>
    </Card>
    <div className="space-y-5" aria-live="polite" aria-busy={busy}>
    {result?<>
     <Card className="border-primary/20 bg-[#eaf3ec] shadow-none"><CardContent><div className="mb-5 flex items-center justify-between"><span className="text-xs font-semibold uppercase tracking-widest text-primary">02 / Prediction</span><Badge variant="outline" className="bg-white/60">Model output</Badge></div><div className="flex items-center gap-5"><span className="flex size-20 items-center justify-center rounded-2xl bg-primary text-5xl font-medium text-white">{result.prediction}</span><div><h2 className="text-2xl font-semibold">Category {result.prediction}</h2><p className="mt-1 text-primary">{titles[result.prediction]}</p></div></div><p className="mt-5 text-sm leading-relaxed">{result.explanation.summary}</p>{stale&&<p role="status" className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">Inputs changed. Predict again to update this result.</p>}</CardContent></Card>
     <Card className="shadow-none"><CardHeader><CardTitle className="flex items-center gap-2 text-lg"><Sparkles size={18}/> Why this category?</CardTitle><CardDescription>Plain language first. The original model details are one tab away.</CardDescription></CardHeader><CardContent><Tabs defaultValue="plain"><TabsList className="mb-5 w-full"><TabsTrigger value="plain" className="flex-1">Simple explanation</TabsTrigger><TabsTrigger value="technical" className="flex-1">Model details</TabsTrigger></TabsList><TabsContent value="plain"><ol className="space-y-4">{result.explanation.steps.map((step,i)=><li key={i} className="flex gap-3 text-sm leading-relaxed"><span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-secondary text-xs font-semibold text-primary">{i+1}</span><span>{step}</span></li>)}</ol><div className="mt-5 rounded-lg bg-muted p-4 text-sm leading-relaxed"><p className="mb-2 flex items-center gap-2 font-medium"><Check size={16}/> Where the path ends</p><p>The final branch contains {result.leaf_samples} training profiles. Category {result.prediction} is the most common category among them.</p></div><p className="mt-4 text-xs leading-relaxed text-muted-foreground">Not used on this path: {result.explanation.unused_features.join(', ')||'none'}. These inputs did not decide this prediction.</p></TabsContent><TabsContent value="technical" className="space-y-7"><div><h3 className="mb-3 text-sm font-semibold">Exact decision path</h3><ol className="space-y-2">{result.decision_path.map((step,i)=><li key={i} className="rounded-md bg-muted px-3 py-2 font-mono text-xs">{i+1}. {step}</li>)}</ol></div><div><h3 className="mb-4 text-sm font-semibold">Category proportions in the final leaf</h3><Bars values={result.probabilities} categories/><p className="mt-3 text-xs text-muted-foreground">{result.probability_note}</p></div><div><h3 className="mb-4 text-sm font-semibold">Local feature importance · this path</h3><Bars values={result.local_feature_contributions}/><p className="mt-3 text-xs text-muted-foreground">{result.local_importance_note}</p></div><div><h3 className="mb-4 text-sm font-semibold">Global feature importance · whole model</h3><Bars values={result.global_feature_importance}/></div></TabsContent></Tabs></CardContent></Card>
    </>:<Card className="min-h-80 border-dashed bg-white/50 shadow-none"><CardContent className="flex min-h-72 flex-col items-center justify-center text-center"><div className="mb-5 rounded-2xl border bg-white p-4 text-primary"><GitBranch size={30}/></div><h2 className="text-xl font-semibold">See the reasoning, not just a label.</h2><p className="mt-3 max-w-xs text-sm leading-relaxed text-muted-foreground">Your predicted category, simple explanation and full decision path will appear here.</p><span className="mt-6 flex items-center gap-2 text-xs text-primary"><ArrowRight size={14}/> Fill in the profile and predict</span></CardContent></Card>}
    </div>
   </div>
  </main>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);
