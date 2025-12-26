# Animations & Visual Effects Demo

This page demonstrates all the visual enhancements and animations implemented in Phase 3.


## 🎬 Keyframe Animations

### Fade Effects

<div class="animate-fadeIn">
<strong>fadeIn:</strong> This element fades in smoothly.
</div>

<div class="animate-fadeInUp delay-200">
<strong>fadeInUp:</strong> This element fades in while sliding up from below.
</div>

<div class="animate-fadeInDown delay-300">
<strong>fadeInDown:</strong> This element fades in while sliding down from above.
</div>

### Slide Effects

<div class="animate-slideInLeft delay-200">
<strong>slideInLeft:</strong> This element slides in from the left.
</div>

<div class="animate-slideInRight delay-300">
<strong>slideInRight:</strong> This element slides in from the right.
</div>

### Scale & Motion

<div class="animate-scaleIn delay-200">
<strong>scaleIn:</strong> This element scales up from 95% to 100%.
</div>

<div class="animate-bounce" style="display: inline-block;">
🎾 <strong>bounce:</strong> Bouncing animation
</div>

<div class="animate-pulse" style="display: inline-block; margin-left: 2rem;">
💓 <strong>pulse:</strong> Pulsing animation
</div>


## 🔗 Interactive Link Effects

Hover over these links to see the animated underline effect:

- [Regular internal link](../built_in_graders/overview.md)
- [External link with icon animation](https://example.com)
- Link with code inside: [`openjudge.graders`](../built_in_graders/overview.md)


## 💻 Code Block Enhancements

Hover over the code block to see the shadow lift effect and the copy button appear:

```python
from openjudge import GraderRunner

# Initialize grader
runner = GraderRunner()

# Run evaluation
results = runner.run(dataset="examples")
print(f"Results: {results}")
```

### Multiple Languages with Tabs

=== "Python"

    ```python
    def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    # Usage
    message = greet("OpenJudge")
    print(message)
    ```

=== "TypeScript"

    ```typescript
    function greet(name: string): string {
      // Greet a person by name
      return `Hello, ${name}!`;
    }

    // Usage
    const message = greet("OpenJudge");
    console.log(message);
    ```

=== "Bash"

    ```bash
    #!/bin/bash

    # Greet a person by name
    greet() {
      echo "Hello, $1!"
    }

    # Usage
    greet "OpenJudge"
    ```


## 📦 Card & Container Effects

Hover over these admonitions to see the lift effect:

!!! note "Interactive Note"
    This note card will lift slightly when you hover over it. The shadow also becomes more pronounced, creating a sense of depth.

!!! tip "Helpful Tip with Animation"
    This tip box demonstrates the same hover effect. Try hovering over different parts of the page to see consistent interactions.

!!! warning "Important Warning"
    Warnings also have the hover effect. This creates a cohesive design language across all UI elements.

!!! danger "Critical Alert"
    Even danger alerts have the smooth hover animation, maintaining visual consistency.


## 📋 Collapsible Sections

Click to expand these collapsible sections and watch the smooth animation:

??? note "Click to expand - Animation Demo"
    The content smoothly slides down when you click the header.

    Features:
    - Smooth expand/collapse animation
    - Arrow rotation
    - Background color change on hover

    ```python
    # Even code blocks inside collapsibles are animated
    print("Smooth animations!")
    ```

???+ tip "Expanded by Default"
    This section starts expanded, but you can collapse it to see the smooth animation in reverse.

    The animation system uses CSS transitions and keyframes for optimal performance.


## 🖼️ Image Effects

Hover over images to see the zoom and shadow effect:

![Example Image](https://via.placeholder.com/600x300/3b82f6/ffffff?text=Hover+over+me)

*Note: Images also have lazy loading with shimmer animation while loading*


## 📊 Table Hover Effects

Hover over table rows to see the highlight effect:

| Grader Type | Animation | Description |
|-------------|-----------|-------------|
| LLM Grader | Row Hover | Smooth background color transition |
| Code Grader | Shadow | Subtle shadow on hover |
| Text Grader | Transform | Slight lift effect |
| Agent Grader | Combined | Multiple effects working together |


## 🔄 Workflow Step Animations

Hover over workflow steps to see the transform effect:

<div class="workflow-single">
<div class="workflow-header">Animated Workflow Example</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>First Step</strong>

Hover over this step to see it shift to the right and the number badge scale up.</li>

<li><strong>Second Step</strong>

Each step has independent hover states with smooth transitions.</li>

<li><strong>Third Step</strong>

The animation uses hardware acceleration for smooth performance.</li>

<li><strong>Final Step</strong>

All animations respect the user's reduced motion preferences.</li>
</ol>
</div>
</div>


## 🎨 Loading States

### Spinner

<div class="spinner"></div> Loading...

### Skeleton Loader

<div class="skeleton" style="width: 100%; height: 60px; margin: 1rem 0;"></div>
<div class="skeleton" style="width: 80%; height: 40px; margin: 1rem 0;"></div>
<div class="skeleton" style="width: 90%; height: 40px;"></div>


## ✨ Special Effects

### Gradient Divider

Content above divider

<hr class="gradient">

Content below divider

### Decorative Divider

Standard content


More content with standard divider


## 🎯 Scroll-Triggered Animations

Scroll down to see elements animate into view:

<div class="fade-in-on-scroll">
### This Heading Fades In

This entire section will fade in as you scroll to it.
</div>

<div class="slide-in-left">
### Slide from Left

This section slides in from the left side of the screen.
</div>

<div class="slide-in-right">
### Slide from Right

This section slides in from the right side of the screen.
</div>


## 🔔 Notification Styles

### Success (Bounce In)

<div class="bounce-in" style="padding: 1rem; background: #d1fae5; border: 1px solid #10b981; border-radius: 0.5rem; margin: 1rem 0;">
✓ <strong>Success!</strong> Your action was completed successfully.
</div>

### Error (Shake)

<div class="shake" style="padding: 1rem; background: #fee2e2; border: 1px solid #ef4444; border-radius: 0.5rem; margin: 1rem 0;">
✗ <strong>Error!</strong> Something went wrong. Please try again.
</div>


## ⚙️ Animation Utilities

You can use these utility classes in your Markdown:

| Class | Effect | Usage |
|-------|--------|-------|
| `.animate-fadeIn` | Fade in | `<div class="animate-fadeIn">...</div>` |
| `.animate-fadeInUp` | Fade in + slide up | `<div class="animate-fadeInUp">...</div>` |
| `.animate-slideInLeft` | Slide from left | `<div class="animate-slideInLeft">...</div>` |
| `.animate-pulse` | Pulse effect | `<span class="animate-pulse">...</span>` |
| `.animate-spin` | Spin animation | `<div class="animate-spin">⚙️</div>` |
| `.delay-100` | 100ms delay | Combine with other classes |
| `.delay-200` | 200ms delay | Combine with other classes |
| `.duration-fast` | Fast animation | Override default duration |

### Combining Utilities

<div class="animate-fadeInUp delay-300 duration-slow">
This element fades in and slides up with a 300ms delay and slow duration.
</div>


## 🎛️ Accessibility

All animations respect the user's motion preferences:

- Users who prefer reduced motion will see instant transitions
- Focus indicators are enhanced with smooth animations
- All interactive elements have proper focus states
- Animations don't interfere with screen readers

To test: Enable "Reduce Motion" in your system preferences and reload this page.


## 🏗️ Performance Notes

All animations are optimized for performance:

- ✅ Hardware acceleration enabled
- ✅ Animations use `transform` and `opacity` (GPU-accelerated)
- ✅ No layout shifts during animations
- ✅ Debounced scroll handlers
- ✅ Intersection Observer for scroll animations
- ✅ Minimal repaints and reflows


## 📚 Implementation Details

These animations are implemented using:

1. **CSS Keyframes** - For reusable animation definitions
2. **CSS Transitions** - For hover and state changes
3. **CSS Variables** - For consistent timing and easing
4. **JavaScript** - For scroll-triggered and interactive animations
5. **Intersection Observer API** - For efficient scroll detection

All code is located in:
- `/docs/stylesheets/animations.css` - All CSS animations
- `/docs/javascripts/animations.js` - JavaScript enhancements


## 🎨 Customization

You can customize animations by modifying CSS variables:

```css
:root {
  --rm-transition-fast: 0.15s;    /* Quick interactions */
  --rm-transition-normal: 0.25s;  /* Standard transitions */
  --rm-transition-slow: 0.4s;     /* Slower, more dramatic */
  --rm-ease-smooth: cubic-bezier(0.4, 0, 0.2, 1);
}
```


<div style="text-align: center; margin: 3rem 0;">
<div class="animate-pulse">
✨ <strong>Phase 3 Complete!</strong> ✨
</div>
</div>

