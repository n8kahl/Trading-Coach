"use client";

import { useEffect } from "react";

import { Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from "@/components/ui/toast";
import { ToastProviderCustom, useToast } from "@/components/ui/use-toast";

export function Toaster() {
  return (
    <ToastProviderCustom>
      <ToastProvider>
        <ToastViewport />
        <ToastListener />
      </ToastProvider>
    </ToastProviderCustom>
  );
}

function ToastListener() {
  const { toasts, dismiss } = useToast();

  useEffect(() => {
    const timers = toasts.map((toast) => {
      if (toast.duration === Infinity) return undefined;
      return setTimeout(() => dismiss(toast.id), toast.duration ?? 3500);
    });
    return () => {
      timers.forEach((timer) => timer && clearTimeout(timer));
    };
  }, [toasts, dismiss]);

  return (
    <>
      {toasts.map((toast) => (
        <Toast key={toast.id} variant={toast.variant} onOpenChange={(open) => !open && dismiss(toast.id)}>
          <div className="grid gap-1">
            {toast.title && <ToastTitle>{toast.title}</ToastTitle>}
            {toast.description && <ToastDescription>{toast.description}</ToastDescription>}
          </div>
          <ToastClose />
        </Toast>
      ))}
    </>
  );
}
