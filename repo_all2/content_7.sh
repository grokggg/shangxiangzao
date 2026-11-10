#!/usr/bin/env bash

set -euo pipefail

export PATH="/nix/runtime/nix-env/bin:$PATH"
export FONTCONFIG_FILE=/nix/runtime/nix-env/etc/fonts/fonts.conf
export LD_LIBRARY_PATH=/nix/runtime/nix-env/lib
export NEXUS_GUI_DISPLAY="${NEXUS_GUI_DISPLAY:-:99}"
export NEXUS_GUI_SCREEN_WIDTH="${NEXUS_GUI_SCREEN_WIDTH:-1440}"
export NEXUS_GUI_SCREEN_HEIGHT="${NEXUS_GUI_SCREEN_HEIGHT:-900}"
export NEXUS_GUI_SCREEN_DEPTH="${NEXUS_GUI_SCREEN_DEPTH:-24}"
export NEXUS_GUI_XVFB_COMMAND="${NEXUS_GUI_XVFB_COMMAND:-}"
export NEXUS_GUI_VNC_HOST="${NEXUS_GUI_VNC_HOST:-127.0.0.1}"
export NEXUS_GUI_VNC_PORT="${NEXUS_GUI_VNC_PORT:-5900}"
export NEXUS_GUI_NOVNC_HOST="${NEXUS_GUI_NOVNC_HOST:-0.0.0.0}"
export NEXUS_GUI_NOVNC_PORT="${NEXUS_GUI_NOVNC_PORT:-6080}"
export NEXUS_GUI_WINDOW_MANAGER="${NEXUS_GUI_WINDOW_MANAGER:-none}"
export NEXUS_GUI_DESKTOP_SESSION="${NEXUS_GUI_DESKTOP_SESSION:-}"
export NEXUS_GUI_GNOME_SESSION_NAME="${NEXUS_GUI_GNOME_SESSION_NAME:-gnome}"
export NEXUS_GUI_APPLY_UBUNTU_THEME="${NEXUS_GUI_APPLY_UBUNTU_THEME:-1}"
export NEXUS_GUI_SESSION_COMMAND="${NEXUS_GUI_SESSION_COMMAND:-}"
export NEXUS_GUI_SESSION_USER="${NEXUS_GUI_SESSION_USER:-}"
export NEXUS_GUI_REQUIRE_SYSTEM_DBUS="${NEXUS_GUI_REQUIRE_SYSTEM_DBUS:-0}"
export DISPLAY="$NEXUS_GUI_DISPLAY"
export XAUTHORITY="${XAUTHORITY:-/tmp/nexus.Xauthority}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/nexus-xdg-runtime}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"
export GDK_BACKEND="${GDK_BACKEND:-x11}"
export GSK_RENDERER="${GSK_RENDERER:-cairo}"
export CLUTTER_BACKEND="${CLUTTER_BACKEND:-x11}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"

touch "$XAUTHORITY"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

gui_novnc_ready() {
    curl -fsS "http://127.0.0.1:${NEXUS_GUI_NOVNC_PORT}/vnc.html" >/dev/null 2>&1
}

start_system_dbus() {
    if [[ -S /run/dbus/system_bus_socket ]]; then
        return 0
    fi
    if ! command -v dbus-daemon >/dev/null 2>&1; then
        echo "dbus-daemon not found; skipping system D-Bus startup"
        if [[ "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-shell" || "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-session" || "$NEXUS_GUI_REQUIRE_SYSTEM_DBUS" == "1" ]]; then
            return 1
        fi
        return 0
    fi

    mkdir -p /run/dbus
    echo "Starting system D-Bus..."
    if dbus-daemon --system --fork --nopidfile; then
        return 0
    fi

    echo "System D-Bus failed to start." >&2
    if [[ "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-shell" || "$NEXUS_GUI_DESKTOP_SESSION" == "gnome-session" || "$NEXUS_GUI_REQUIRE_SYSTEM_DBUS" == "1" ]]; then
        return 1
    fi
    return 0
}

start_optional_service() {
    local name="$1"
    local command="$2"
    local log_file="$3"

    echo "Starting optional GNOME service ${name}: ${command}"
    /nix/runtime/nix-env/bin/bash -lc "$command" >"$log_file" 2>&1 &
}

start_gnome_system_services() {
    if [[ "$NEXUS_GUI_DE