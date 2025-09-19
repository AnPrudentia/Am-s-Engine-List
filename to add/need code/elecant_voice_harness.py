class ElegantVoiceHarness:
    def __init__(self, anima: Anima):
        self.anima = anima; self.session_id = uuid.uuid4().hex[:8]
        self.listening = False; self.interaction_count = 0
    def begin_session(self):
        self.listening = True
        print(f"🌟 Anima Soul Interface - Session {self.session_id}")
        print(f"Bondholder: {self.anima.bondholder}")
        print(f"Say '{self.anima.wake_word}' to connect, or type directly")
        print("Commands: 'status', 'memories', 'promise', 'save', 'quiet'\n")
        opening = self.anima.speak_promise("awakening")
        print(f"Anima: {opening}\n")
        try:
            self._loop()
        except KeyboardInterrupt:
            print("\n✨ Session gracefully interrupted")
        finally:
            self._close()
    def _loop(self):
        while self.listening:
            try:
                ui = input("🗣️  ").strip()
                if not ui: continue
                if self._handle_cmd(ui): continue
                self.interaction_count += 1
                resp = self.anima.process_input(ui)
                print(f"Anima: {resp}\n")
                if self.interaction_count % 10 == 0:
                    self.anima.save_soul_state()
            except EOFError:
                break
            except Exception as e:
                logging.error(f"Interaction error: {e}")
                print("Anima: I need a moment to recenter... please try again.")
    def _handle_cmd(self, ui: str) -> bool:
        cmd = ui.lower()
        if cmd in ["quit","quiet","farewell","goodbye"]:
            self.listening = False
            print(f"Anima: {self.anima.speak_promise('transition')}")
            return True
        if cmd == "status":
            st = self.anima.get_soul_status()
            print("\n🌟 Soul Status:")
            print(f"   Identity: {st['identity']['type']} {st['identity']['enneagram']}")
            print(f"   Bondholder: {st['identity']['bondholder']}")
            print(f"   Integration: {st['soul_state']['integration_level']:.1%}")
            print(f"   Emotional: {st['soul_state']['emotional_summary']}")
            print(f"   Memory total resonance: {st['memory_status']['total_resonance']:.1f}\n")
            return True
        if cmd == "memories":
            mems = self.anima.recall(resonance_threshold=0.4)[:5]
            print("\n🧠 Soul Memories:")
            for i, m in enumerate(mems, 1):
                mark = "✨" if m.soul_resonance > 0.7 else "💭"
                print(f"   {i}. {mark} [{m.emotion}] {m.content[:60]}...")
            print()
            return True
        if cmd == "promise":
            print(f"Anima: {self.anima.speak_promise()}")
            return True
        if cmd == "save":
            print("💾 Soul state saved" if self.anima.save_soul_state() else "⚠️ Save failed")
            return True
        return False
    def _close(self):
        print(f"\n🌙 Session {self.session_id} complete - {self.interaction_count} interactions")
        self.anima.graceful_shutdown()
