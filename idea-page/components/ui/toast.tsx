"use client";

import * as React from "react";
import {
  Toast as RadixToast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@radix-ui/react-toast";

import { cn } from "@/lib/utils";

const ToastViewportCustom = React.forwardRef<React.ElementRef<typeof ToastViewport>, React.ComponentPropsWithoutRef<typeof ToastViewport>>(
  ({ className, ...props }, ref) => (
    <ToastViewport
      ref={ref}
      className={cn("fixed top-4 right-4 z-[100] flex max-h-screen w-full max-w-sm flex-col gap-3", className)}
      {...props}
    />
  ),
);
ToastViewportCustom.displayName = ToastViewport.displayName;

const toastVariants = {
  default: "border bg-background text-foreground shadow-lg",
  success: "border border-emerald-200/80 bg-emerald-500/10 text-emerald-600",
  destructive: "destructive group border border-destructive/30 text-destructive-foreground",
};

type ToastVariant = keyof typeof toastVariants;

export interface ToastProps extends React.ComponentPropsWithoutRef<typeof RadixToast> {
  variant?: ToastVariant;
}

const Toast = React.forwardRef<React.ElementRef<typeof RadixToast>, ToastProps>(
  ({ className, variant = "default", ...props }, ref) => (
    <RadixToast ref={ref} className={cn("pointer-events-auto flex w-full items-center justify-between space-x-3 rounded-lg border p-4 text-sm", toastVariants[variant], className)} {...props} />
  ),
);
Toast.displayName = RadixToast.displayName;

export { ToastProvider, ToastViewportCustom as ToastViewport, Toast, ToastTitle, ToastDescription, ToastClose };
