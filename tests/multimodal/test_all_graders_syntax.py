# -*- coding: utf-8 -*-
"""
Syntax check for all multimodal graders

This script imports all graders to verify there are no syntax or import errors.
"""

import sys


def test_imports():
    """Test that all graders can be imported without errors"""
    print("=" * 80)
    print("Testing Multimodal Graders Import")
    print("=" * 80)

    errors = []

    # Test ImageCoherenceGrader
    print("\n1. Testing ImageCoherenceGrader...")
    try:
        from rm_gallery.gallery.grader.multimodal.image_coherence import (
            ImageCoherenceGrader,
        )

        print("   ✓ ImageCoherenceGrader imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import ImageCoherenceGrader: {e}")
        errors.append(("ImageCoherenceGrader", str(e)))

    # Test ImageHelpfulnessGrader
    print("\n2. Testing ImageHelpfulnessGrader...")
    try:
        from rm_gallery.gallery.grader.multimodal.image_helpfulness import (
            ImageHelpfulnessGrader,
        )

        print("   ✓ ImageHelpfulnessGrader imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import ImageHelpfulnessGrader: {e}")
        errors.append(("ImageHelpfulnessGrader", str(e)))

    # Test ImageReferenceGrader
    print("\n3. Testing ImageReferenceGrader...")
    try:
        from rm_gallery.gallery.grader.multimodal.image_reference import (
            ImageReferenceGrader,
        )

        print("   ✓ ImageReferenceGrader imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import ImageReferenceGrader: {e}")
        errors.append(("ImageReferenceGrader", str(e)))

    # Test ImageEditingGrader
    print("\n4. Testing ImageEditingGrader...")
    try:
        from rm_gallery.gallery.grader.multimodal.image_editing import (
            ImageEditingGrader,
        )

        print("   ✓ ImageEditingGrader imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import ImageEditingGrader: {e}")
        errors.append(("ImageEditingGrader", str(e)))

    # Test TextToImageGrader
    print("\n5. Testing TextToImageGrader...")
    try:
        from rm_gallery.gallery.grader.multimodal.text_to_image import TextToImageGrader

        print("   ✓ TextToImageGrader imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import TextToImageGrader: {e}")
        errors.append(("TextToImageGrader", str(e)))

    # Test MultimodalGEvalGrader
    print("\n6. Testing MultimodalGEvalGrader...")
    try:
        from rm_gallery.gallery.grader.multimodal.multimodal_geval import (
            MultimodalGEvalGrader,
        )

        print("   ✓ MultimodalGEvalGrader imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import MultimodalGEvalGrader: {e}")
        errors.append(("MultimodalGEvalGrader", str(e)))

    # Summary
    print("\n" + "=" * 80)
    if errors:
        print(f"FAILED: {len(errors)} grader(s) failed to import")
        print("=" * 80)
        for name, error in errors:
            print(f"\n{name}:")
            print(f"  {error}")
        return 1
    else:
        print("SUCCESS: All graders imported successfully!")
        print("=" * 80)
        return 0


def test_grader_signatures():
    """Test that evaluate methods have correct signatures"""
    print("\n" + "=" * 80)
    print("Testing Grader Method Signatures")
    print("=" * 80)

    import inspect

    # Import all graders
    from rm_gallery.gallery.grader.multimodal.image_coherence import (
        ImageCoherenceGrader,
    )
    from rm_gallery.gallery.grader.multimodal.image_editing import ImageEditingGrader
    from rm_gallery.gallery.grader.multimodal.image_helpfulness import (
        ImageHelpfulnessGrader,
    )
    from rm_gallery.gallery.grader.multimodal.image_reference import (
        ImageReferenceGrader,
    )
    from rm_gallery.gallery.grader.multimodal.multimodal_geval import (
        MultimodalGEvalGrader,
    )
    from rm_gallery.gallery.grader.multimodal.text_to_image import TextToImageGrader

    graders = [
        ("ImageCoherenceGrader", ImageCoherenceGrader),
        ("ImageHelpfulnessGrader", ImageHelpfulnessGrader),
        ("ImageReferenceGrader", ImageReferenceGrader),
        ("ImageEditingGrader", ImageEditingGrader),
        ("TextToImageGrader", TextToImageGrader),
        ("MultimodalGEvalGrader", MultimodalGEvalGrader),
    ]

    errors = []

    for name, grader_class in graders:
        print(f"\n{name}:")

        # Check if evaluate method exists
        if not hasattr(grader_class, "evaluate"):
            print(f"  ✗ No evaluate method found")
            errors.append((name, "Missing evaluate method"))
            continue

        # Get evaluate method signature
        sig = inspect.signature(grader_class.evaluate)
        params = list(sig.parameters.keys())

        # Check if async_mode parameter exists (it shouldn't)
        if "async_mode" in params:
            print(f"  ✗ async_mode parameter found (should be removed)")
            errors.append((name, "async_mode parameter not removed"))
        else:
            print(f"  ✓ async_mode parameter correctly removed")

        # Check if method is async
        if inspect.iscoroutinefunction(grader_class.evaluate):
            print(f"  ✓ evaluate method is async")
        else:
            print(f"  ✗ evaluate method is not async")
            errors.append((name, "evaluate method not async"))

    print("\n" + "=" * 80)
    if errors:
        print(f"FAILED: {len(errors)} issue(s) found")
        for name, error in errors:
            print(f"  {name}: {error}")
        return 1
    else:
        print("SUCCESS: All method signatures are correct!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    result1 = test_imports()
    if result1 == 0:
        result2 = test_grader_signatures()
        sys.exit(result2)
    else:
        sys.exit(result1)
