"use client";
import { useSearchParams } from 'next/navigation';
import { useEffect } from 'react';
import { setRuntimeApiBase } from '@/lib/runtimeTarget';

export default function ClientRuntimeSetter() {
  const params = useSearchParams();
  const server = params.get('server') || undefined;
  useEffect(() => { if (server) setRuntimeApiBase(server); }, [server]);
  return null;
}

