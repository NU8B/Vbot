"""
Welcome Screen Module for Vbot
Provides avatar selection and recommendation system for new users
"""

import tkinter as tk
from tkinter import ttk
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Callable
from .user_preferences import (
    get_user_preferences,
    record_avatar_selection,
    get_user_profile_for_recommendation,
)

# Optional PIL import for future image support
try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class AvatarRecommender:
    """Handles avatar recommendation logic based on user preferences"""

    def __init__(self):
        self.avatar_profiles = {
            "Amelia": {
                "name": "Amelia Watson",
                "personality": "Detective, Curious, Energetic",
                "voice_type": "Bright and cheerful",
                "gender": "Female",
                "description": "A time-traveling detective with boundless curiosity and energy. Perfect for users who enjoy mystery and adventure.",
                "tags": ["detective", "energetic", "curious", "adventure"],
                "color": "#ffd05c",
            },
            "Eveland": {
                "name": "Eveland Novice",
                "personality": "Calm, Intellectual, Sophisticated",
                "voice_type": "Deep and thoughtful",
                "gender": "Male",
                "description": "A sophisticated and intellectual companion who brings wisdom and calm to conversations.",
                "tags": ["intellectual", "calm", "sophisticated", "wisdom"],
                "color": "#318fc5",
            },
            "Gura": {
                "name": "Gawr Gura",
                "personality": "Playful, Friendly, Mischievous",
                "voice_type": "Cute and playful",
                "gender": "Female",
                "description": "A friendly shark with a playful personality. Great for casual conversations and fun interactions.",
                "tags": ["playful", "friendly", "cute", "casual"],
                "color": "#4a90e2",
            },
            "Shiori": {
                "name": "Shiori Novella",
                "personality": "Mysterious, Artistic, Thoughtful",
                "voice_type": "Soft and mysterious",
                "gender": "Female",
                "description": "An artistic and mysterious companion who brings depth and creativity to your conversations.",
                "tags": ["mysterious", "artistic", "creative", "thoughtful"],
                "color": "#85318c",
            },
            "Wilson": {
                "name": "Wilson",
                "personality": "Reliable, Supportive, Steady",
                "voice_type": "Warm and steady",
                "gender": "Male",
                "description": "A reliable and supportive companion who provides steady guidance and warm conversations.",
                "tags": ["reliable", "supportive", "steady", "warm"],
                "color": "#8eeefe",
            },
        }

    def get_avatar_info(self, avatar_name: str) -> Dict:
        """Get detailed information about an avatar"""
        return self.avatar_profiles.get(avatar_name, {})

    def get_all_avatars(self) -> List[str]:
        """Get list of all available avatars"""
        return list(self.avatar_profiles.keys())

    def recommend_avatar(self, preferences: Dict) -> str:
        """Recommend an avatar based on user preferences"""
        # Enhanced recommendation logic with better balance
        preferred_gender = preferences.get("gender", "").lower()
        preferred_personality = preferences.get("personality", [])
        favorite_avatar = preferences.get("favorite_avatar")
        selection_history = preferences.get("selection_history", [])
        
        # Check if this is a fresh recommendation (no explicit preferences)
        has_explicit_preferences = bool(preferred_gender or preferred_personality)

        scores = {}
        for avatar_name, profile in self.avatar_profiles.items():
            score = 0

            # Gender preference (STRONG weight: 15) - This should be the primary factor
            if preferred_gender and profile["gender"].lower() == preferred_gender:
                score += 15
            # Gender mismatch penalty - strongly discourage wrong gender
            elif preferred_gender and profile["gender"].lower() != preferred_gender:
                score -= 8

            # Personality tags matching - improved logic to handle contradictions
            avatar_tags = profile["tags"]
            personality_matches = 0
            for pref in preferred_personality:
                pref_lower = pref.lower()
                # Check if preference matches any tag or is in the personality/description
                if pref_lower in [tag.lower() for tag in avatar_tags]:
                    score += 3
                    personality_matches += 1
                elif pref_lower in profile["personality"].lower():
                    score += 2
                    personality_matches += 1
                elif pref_lower in profile["description"].lower():
                    score += 1
                    personality_matches += 1

            # Apply personality match bonus for good fits
            if personality_matches >= 2:
                score += 2  # Bonus for multiple personality matches

            # Favorite avatar bonus (reduced weight and only when no explicit preferences)
            if favorite_avatar and avatar_name == favorite_avatar:
                if has_explicit_preferences:
                    score += 1  # Minimal bonus when user has explicit preferences
                else:
                    score += 2  # Larger bonus only when no explicit preferences

            # Recent usage penalty (to encourage variety)
            recent_selections = [record["avatar"] for record in selection_history[-3:]]
            if avatar_name in recent_selections:
                score -= 1

            scores[avatar_name] = score

        # Return avatar with highest score
        if scores:
            # Sort by score (descending), then by name for consistency
            sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            top_score = sorted_scores[0][1]
            
            # If all scores are very low or negative, use better fallback logic
            if top_score <= 0:
                if preferred_gender:
                    # Find the best gender match
                    gender_matches = [
                        (name, score)
                        for name, score in scores.items()
                        if self.avatar_profiles[name]["gender"].lower() == preferred_gender
                    ]
                    if gender_matches:
                        best_gender_match = max(gender_matches, key=lambda x: (x[1], x[0]))
                        return best_gender_match[0]
                
                # If no gender preference or no gender matches, use balanced fallback
                return self._get_balanced_fallback(selection_history)
            
            return sorted_scores[0][0]

        # Absolute fallback (should never happen)
        return self._get_balanced_fallback(selection_history)
        
    def _get_balanced_fallback(self, selection_history: list) -> str:
        """Get a balanced fallback avatar that hasn't been used recently"""
        # Get avatars sorted by least recent usage
        recent_avatars = [record["avatar"] for record in selection_history[-5:]]
        
        # Find avatars not used recently
        all_avatars = list(self.avatar_profiles.keys())
        unused_avatars = [avatar for avatar in all_avatars if avatar not in recent_avatars]
        
        if unused_avatars:
            # Return first unused avatar alphabetically for consistency
            return sorted(unused_avatars)[0]
        else:
            # If all avatars have been used recently, return the least recently used
            if selection_history:
                used_avatars = [record["avatar"] for record in selection_history]
                # Count usage frequency
                avatar_counts = {}
                for avatar in used_avatars:
                    avatar_counts[avatar] = avatar_counts.get(avatar, 0) + 1
                
                # Return least used avatar
                least_used = min(avatar_counts.items(), key=lambda x: (x[1], x[0]))
                return least_used[0]
            
            # Final fallback - return Amelia as a neutral choice
            return "Amelia"


class WelcomeScreen:
    """Main welcome screen interface for avatar selection"""

    def __init__(self, on_avatar_selected: Callable[[str], None]):
        self.on_avatar_selected = on_avatar_selected
        self.recommender = AvatarRecommender()
        self.selected_avatar = None
        self.root = None
        self.avatar_frames = {}
        self.avatar_images = {}
        self.user_prefs = get_user_preferences()

        # Pre-select last used avatar if available
        last_avatar = self.user_prefs.get_last_selected_avatar()
        if last_avatar and last_avatar in self.recommender.get_all_avatars():
            self.selected_avatar = last_avatar

    def create_window(self) -> tk.Tk:
        """Create and configure the welcome screen window"""
        self.root = tk.Tk()
        self.root.title("Welcome to Vbot - Choose Your AI Companion")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")

        self._create_header()
        self._create_avatar_selection()
        self._create_footer()

        return self.root

    def _create_header(self):
        """Create the header section with title and subtitle"""
        header_frame = tk.Frame(self.root, bg="#1a1a2e", height=120)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)

        # Main title
        title_label = tk.Label(
            header_frame,
            text="Welcome to Vbot",
            font=("Arial", 32, "bold"),
            fg="#ffffff",
            bg="#1a1a2e",
        )
        title_label.pack(pady=(20, 5))

        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Choose your AI companion to get started",
            font=("Arial", 16),
            fg="#b0b0b0",
            bg="#1a1a2e",
        )
        subtitle_label.pack()

    def _create_avatar_selection(self):
        """Create the avatar selection grid"""
        # Main content frame
        content_frame = tk.Frame(self.root, bg="#1a1a2e")
        content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Create scrollable frame
        canvas = tk.Canvas(content_frame, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            content_frame, orient="vertical", command=canvas.yview
        )
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Grid layout for avatars
        avatars = self.recommender.get_all_avatars()
        cols = 3  # 3 avatars per row

        for i, avatar_name in enumerate(avatars):
            row = i // cols
            col = i % cols

            self._create_avatar_card(scrollable_frame, avatar_name, row, col)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _create_avatar_card(self, parent, avatar_name: str, row: int, col: int):
        """Create an individual avatar selection card"""
        avatar_info = self.recommender.get_avatar_info(avatar_name)

        # Main card frame
        card_frame = tk.Frame(
            parent, bg="#2b2b3b", relief=tk.RAISED, borderwidth=2, padx=15, pady=15
        )
        card_frame.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

        # Configure grid weights for responsiveness
        parent.grid_columnconfigure(col, weight=1)
        parent.grid_rowconfigure(row, weight=1)

        # Store frame reference for selection highlighting
        self.avatar_frames[avatar_name] = card_frame

        # Avatar image placeholder
        image_frame = tk.Frame(card_frame, bg="#3b3b4b", width=150, height=150)
        image_frame.pack(pady=(0, 10))
        image_frame.pack_propagate(False)

        # Placeholder image (will be replaced with actual avatar images later)
        placeholder_label = tk.Label(
            image_frame,
            text="👤",
            font=("Arial", 48),
            fg=avatar_info.get("color", "#ffffff"),
            bg="#3b3b4b",
        )
        placeholder_label.pack(expand=True)

        # Avatar name
        name_label = tk.Label(
            card_frame,
            text=avatar_info.get("name", avatar_name),
            font=("Arial", 16, "bold"),
            fg="#ffffff",
            bg="#2b2b3b",
        )
        name_label.pack(pady=(0, 5))

        # Personality
        personality_label = tk.Label(
            card_frame,
            text=avatar_info.get("personality", ""),
            font=("Arial", 11),
            fg=avatar_info.get("color", "#b0b0b0"),
            bg="#2b2b3b",
        )
        personality_label.pack(pady=(0, 5))

        # Voice type
        voice_label = tk.Label(
            card_frame,
            text=f"Voice: {avatar_info.get('voice_type', '')}",
            font=("Arial", 10),
            fg="#b0b0b0",
            bg="#2b2b3b",
        )
        voice_label.pack(pady=(0, 10))

        # Description
        desc_label = tk.Label(
            card_frame,
            text=avatar_info.get("description", ""),
            font=("Arial", 10),
            fg="#d0d0d0",
            bg="#2b2b3b",
            wraplength=200,
            justify=tk.LEFT,
        )
        desc_label.pack(pady=(0, 15))

        # Select button
        select_btn = tk.Button(
            card_frame,
            text="Select",
            font=("Arial", 12, "bold"),
            bg=avatar_info.get("color", "#4a90e2"),
            fg="black",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=lambda name=avatar_name: self._select_avatar(name),
        )
        select_btn.pack()

        # Hover effects
        def on_enter(
            e,
            frame=card_frame,
            btn=select_btn,
            color=avatar_info.get("color", "#4a90e2"),
        ):
            frame.configure(bg="#3b3b4b")
            # Darken button color on hover
            darker_color = self._darken_color(color)
            btn.configure(bg=darker_color)

        def on_leave(
            e,
            frame=card_frame,
            btn=select_btn,
            color=avatar_info.get("color", "#4a90e2"),
        ):
            if avatar_name != self.selected_avatar:
                frame.configure(bg="#2b2b3b")
            btn.configure(bg=color)

        card_frame.bind("<Enter>", on_enter)
        card_frame.bind("<Leave>", on_leave)
        select_btn.bind("<Enter>", on_enter)
        select_btn.bind("<Leave>", on_leave)

        # Make entire card clickable
        def on_card_click(e, name=avatar_name):
            self._select_avatar(name)

        card_frame.bind("<Button-1>", on_card_click)
        for child in card_frame.winfo_children():
            if isinstance(child, tk.Label):
                child.bind("<Button-1>", on_card_click)

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 20%"""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip("#")
            # Convert to RGB
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            # Darken by 20%
            darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
            # Convert back to hex
            return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
        except:
            return "#333333"  # Fallback color

    def _select_avatar(self, avatar_name: str):
        """Handle avatar selection"""
        # Reset previous selection
        if self.selected_avatar and self.selected_avatar in self.avatar_frames:
            self.avatar_frames[self.selected_avatar].configure(bg="#2b2b3b")

        # Set new selection
        self.selected_avatar = avatar_name
        if avatar_name in self.avatar_frames:
            avatar_info = self.recommender.get_avatar_info(avatar_name)
            # Highlight selected card
            self.avatar_frames[avatar_name].configure(bg="#3b3b4b")

        # Enable continue button
        if hasattr(self, "continue_btn"):
            self.continue_btn.configure(state="normal")

    def _create_footer(self):
        """Create the footer with continue button"""
        footer_frame = tk.Frame(self.root, bg="#1a1a2e", height=80)
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        footer_frame.pack_propagate(False)

        # Continue button
        self.continue_btn = tk.Button(
            footer_frame,
            text="Continue with Selected Avatar",
            font=("Arial", 14, "bold"),
            bg="#4a90e2",
            fg="white",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            state="disabled",
            command=self._continue_to_app,
        )
        self.continue_btn.pack(side=tk.RIGHT, pady=20)

        # Recommendation button
        recommend_btn = tk.Button(
            footer_frame,
            text="Get Recommendation",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            command=self._show_recommendation_dialog,
        )
        recommend_btn.pack(side=tk.LEFT, pady=20)

    def _continue_to_app(self):
        """Continue to main application with selected avatar"""
        if self.selected_avatar:
            # Record the avatar selection in user preferences
            record_avatar_selection(self.selected_avatar)
            self.root.destroy()
            self.on_avatar_selected(self.selected_avatar)

    def _show_recommendation_dialog(self):
        """Show recommendation dialog to help users choose"""
        dialog = RecommendationDialog(self.root, self.recommender, self._select_avatar)
        dialog.show()

    def run(self):
        """Run the welcome screen"""
        self.create_window()
        self.root.mainloop()


class RecommendationDialog:
    """Dialog for getting user preferences and recommending avatars"""

    def __init__(
        self,
        parent,
        recommender: AvatarRecommender,
        on_recommendation: Callable[[str], None],
    ):
        self.parent = parent
        self.recommender = recommender
        self.on_recommendation = on_recommendation
        self.dialog = None
        self.preferences = {}

    def show(self):
        """Show the recommendation dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Get Avatar Recommendation")
        self.dialog.geometry("500x600")  # Increased height
        self.dialog.configure(bg="#1a1a2e")
        self.dialog.resizable(True, True)  # Allow resizing

        # Center dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center on parent with updated dimensions
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - 250
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - 300
        self.dialog.geometry(f"500x600+{x}+{y}")

        self._create_dialog_content()

    def _create_dialog_content(self):
        """Create the dialog content"""
        # Title
        title_label = tk.Label(
            self.dialog,
            text="Tell us about yourself",
            font=("Arial", 18, "bold"),
            fg="#ffffff",
            bg="#1a1a2e",
        )
        title_label.pack(pady=(20, 10))

        # Subtitle
        subtitle_label = tk.Label(
            self.dialog,
            text="We'll recommend the perfect AI companion for you",
            font=("Arial", 12),
            fg="#b0b0b0",
            bg="#1a1a2e",
        )
        subtitle_label.pack(pady=(0, 20))

        # Create scrollable content area
        main_frame = tk.Frame(self.dialog, bg="#1a1a2e")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=(0, 20))

        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_frame, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Content frame (now inside scrollable area)
        content_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        content_frame.pack(expand=True, fill=tk.BOTH, padx=10)

        # Gender preference
        gender_frame = tk.Frame(content_frame, bg="#1a1a2e")
        gender_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            gender_frame,
            text="Preferred companion gender:",
            font=("Arial", 12, "bold"),
            fg="#ffffff",
            bg="#1a1a2e",
        ).pack(anchor="w")

        self.gender_var = tk.StringVar(value="no_preference")
        gender_options = [
            ("No preference", "no_preference"),
            ("Female", "female"),
            ("Male", "male"),
        ]

        for text, value in gender_options:
            tk.Radiobutton(
                gender_frame,
                text=text,
                variable=self.gender_var,
                value=value,
                font=("Arial", 11),
                fg="#d0d0d0",
                bg="#1a1a2e",
                selectcolor="#2d2d44",
                activebackground="#1a1a2e",
                activeforeground="#ffffff",
            ).pack(anchor="w", padx=20)

        # Personality preferences
        personality_frame = tk.Frame(content_frame, bg="#1a1a2e")
        personality_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            personality_frame,
            text="What personality traits do you prefer? (select all that apply)",
            font=("Arial", 12, "bold"),
            fg="#ffffff",
            bg="#1a1a2e",
        ).pack(anchor="w")

        self.personality_vars = {}
        personality_options = [
            "Energetic",
            "Calm",
            "Playful",
            "Intellectual",
            "Mysterious",
            "Friendly",
            "Sophisticated",
            "Reliable",
        ]

        # Create checkboxes in a grid
        checkbox_frame = tk.Frame(personality_frame, bg="#1a1a2e")
        checkbox_frame.pack(fill=tk.X, padx=20, pady=10)

        for i, trait in enumerate(personality_options):
            var = tk.BooleanVar()
            self.personality_vars[trait.lower()] = var

            row = i // 2
            col = i % 2

            tk.Checkbutton(
                checkbox_frame,
                text=trait,
                variable=var,
                font=("Arial", 11),
                fg="#d0d0d0",
                bg="#1a1a2e",
                selectcolor="#2d2d44",
                activebackground="#1a1a2e",
                activeforeground="#ffffff",
            ).grid(row=row, column=col, sticky="w", padx=(0, 20), pady=2)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Buttons (outside scrollable area)
        button_frame = tk.Frame(self.dialog, bg="#1a1a2e")
        button_frame.pack(fill=tk.X, padx=30, pady=(0, 20))

        tk.Button(
            button_frame,
            text="Cancel",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=self.dialog.destroy,
        ).pack(side=tk.LEFT)

        tk.Button(
            button_frame,
            text="Get Recommendation",
            font=("Arial", 12, "bold"),
            bg="#4a90e2",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=self._get_recommendation,
        ).pack(side=tk.RIGHT)

    def _get_recommendation(self):
        """Process preferences and get recommendation"""
        # Collect preferences from dialog
        dialog_preferences = {
            "gender": (
                self.gender_var.get()
                if self.gender_var.get() != "no_preference"
                else ""
            ),
            "personality": [
                trait for trait, var in self.personality_vars.items() if var.get()
            ],
        }

        # Determine if user provided explicit preferences
        has_explicit_preferences = bool(
            dialog_preferences["gender"] or dialog_preferences["personality"]
        )

        if has_explicit_preferences:
            # User provided explicit preferences - use them directly with minimal history influence
            user_profile = get_user_profile_for_recommendation()
            combined_preferences = {
                "gender": dialog_preferences["gender"],
                "personality": dialog_preferences["personality"],
                "interaction_style": user_profile.get("interaction_style", "casual"),
                "favorite_avatar": None,  # Don't use favorite when user has explicit preferences
                "selection_history": user_profile.get("selection_history", [])
            }
        else:
            # No explicit preferences - use existing user profile
            user_profile = get_user_profile_for_recommendation()
            combined_preferences = user_profile

        # Update user profile with new preferences only if they provided some
        if has_explicit_preferences:
            user_prefs = get_user_preferences()
            user_prefs.update_user_profile(dialog_preferences)
            user_prefs.save_preferences()

        # Get recommendation
        recommended_avatar = self.recommender.recommend_avatar(combined_preferences)

        # Show result
        self._show_recommendation_result(recommended_avatar, combined_preferences)

    def _show_recommendation_result(self, recommended_avatar: str, preferences: Dict):
        """Show the recommendation result"""
        # Clear dialog content
        for widget in self.dialog.winfo_children():
            widget.destroy()

        # Result content
        result_frame = tk.Frame(self.dialog, bg="#1a1a2e")
        result_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=30)

        # Title
        tk.Label(
            result_frame,
            text="Perfect Match Found!",
            font=("Arial", 20, "bold"),
            fg="#4CAF50",
            bg="#1a1a2e",
        ).pack(pady=(0, 10))

        # Avatar info
        avatar_info = self.recommender.get_avatar_info(recommended_avatar)

        # Avatar name
        tk.Label(
            result_frame,
            text=avatar_info.get("name", recommended_avatar),
            font=("Arial", 18, "bold"),
            fg=avatar_info.get("color", "#ffffff"),
            bg="#1a1a2e",
        ).pack(pady=(0, 5))

        # Description
        tk.Label(
            result_frame,
            text=avatar_info.get("description", ""),
            font=("Arial", 12),
            fg="#d0d0d0",
            bg="#1a1a2e",
            wraplength=400,
            justify=tk.CENTER,
        ).pack(pady=(0, 20))

        # Personality and voice
        info_frame = tk.Frame(
            result_frame, bg="#2b2b3b", relief=tk.RAISED, borderwidth=1
        )
        info_frame.pack(fill=tk.X, pady=(0, 20), padx=20)

        tk.Label(
            info_frame,
            text=f"Personality: {avatar_info.get('personality', '')}",
            font=("Arial", 11),
            fg="#ffffff",
            bg="#2b2b3b",
        ).pack(pady=5)

        tk.Label(
            info_frame,
            text=f"Voice: {avatar_info.get('voice_type', '')}",
            font=("Arial", 11),
            fg="#b0b0b0",
            bg="#2b2b3b",
        ).pack(pady=5)

        # Buttons
        button_frame = tk.Frame(result_frame, bg="#1a1a2e")
        button_frame.pack(fill=tk.X, pady=(20, 0))

        tk.Button(
            button_frame,
            text="Try Again",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=lambda: (self.dialog.destroy(), self.show()),
        ).pack(side=tk.LEFT)

        tk.Button(
            button_frame,
            text="Select This Avatar",
            font=("Arial", 12, "bold"),
            bg=avatar_info.get("color", "#4a90e2"),
            fg="black",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=lambda: self._select_recommended_avatar(recommended_avatar),
        ).pack(side=tk.RIGHT)

    def _select_recommended_avatar(self, avatar_name: str):
        """Select the recommended avatar and close dialog"""
        self.dialog.destroy()
        self.on_recommendation(avatar_name)


def show_welcome_screen(on_avatar_selected: Callable[[str], None]) -> None:
    """
    Show the welcome screen and call the callback with selected avatar

    Args:
        on_avatar_selected: Callback function that receives the selected avatar name
    """
    welcome = WelcomeScreen(on_avatar_selected)
    welcome.run()


# Example usage and testing
if __name__ == "__main__":

    def on_avatar_selected(avatar_name: str):
        print(f"Selected avatar: {avatar_name}")

    show_welcome_screen(on_avatar_selected)
