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
      {toasts.map(({ id, title, description, variant, onOpenChange, ...toastProps }) => (
        <Toast
          key={id}
          variant={variant}
          {...toastProps}
          onOpenChange={(open) => {
            onOpenChange?.(open);
            if (!open) dismiss(id);
          }}
        >
          <div className="grid gap-1">
            {title && <ToastTitle>{title}</ToastTitle>}
            {description && <ToastDescription>{description}</ToastDescription>}
          </div>
          <ToastClose />
        </Toast>
      ))}
    </>
  );
}
