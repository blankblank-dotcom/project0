# .cursorrules for The Local Titan
- Always prioritize async/await for Reflex State event handlers.
- OpenVINO inference calls must be offloaded to a background thread using `asyncio.to_thread` to prevent WebSocket disconnects.
- Use Pydantic for data validation from VLM outputs.
- Never delete benchmark scripts or OpenVINO configuration headers.