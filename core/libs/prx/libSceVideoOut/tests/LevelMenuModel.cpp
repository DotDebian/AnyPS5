#include "prx/libSceVideoOut/include/LevelMenuModel.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace LevelMenuModel;

namespace {

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "level_menu_model: failed: %s\n", what);
    std::abort();
}

using Strings = std::vector<std::string>;

constexpr const char* SampleXml = R"(<Levels>
	<Category>
		<Id />
		<Name>Meta</Name>
		<LevelType />
		<Title />
		<Level>
			<PlayGoChunk>0</PlayGoChunk>
			<UdsObjectId />
			<Name>Title</Name>
			<File>title_controller_ship</File>
		</Level>
		<Level>
			<Name />
			<File>ps_logo</File>
		</Level>
	</Category>
	<Category>
		<Name>Main Levels</Name>
		<LevelType>Main</LevelType>
		<Level>
			<TotalLeaderboardLevelId />
			<Name>G1 - Aerial Garden</Name>
			<File>underwater_aerial_garden</File>
		</Level>
		<Level>
			<Name>G1 - Snowy Canyon</Name>
			<File> bumper_enemy_snowy_canyon </File>
		</Level>
		<Level>
			<Name>Broken, no file</Name>
			<File />
		</Level>
		<Level>
			<Name>G3 - Casino &amp; Cards</Name>
			<File>timestop_casino</File>
		</Level>
	</Category>
</Levels>
)";

void testParse() {
    const auto entries = ParseLevelList(SampleXml);
    Require(entries.size() == 5, "parse count");
    Require(entries[0] == LevelEntry{"Meta", "Title", "title_controller_ship"}, "first entry");
    Require(entries[1] == LevelEntry{"Meta", "ps_logo", "ps_logo"}, "empty name falls back to file");
    Require(entries[2] == LevelEntry{"Main Levels", "G1 - Aerial Garden", "underwater_aerial_garden"}, "category name ignores LevelType");
    Require(entries[3].file == "bumper_enemy_snowy_canyon", "file is trimmed");
    Require(entries[4].name == "G3 - Casino & Cards", "entities are decoded");
    Require(ParseLevelList("").empty(), "empty document");
    Require(ParseLevelList("<Levels><Category><Name>X</Name><Level><File>a</File>").empty(), "truncated document");
}

void testMoveCategoryToEnd() {
    auto entries = ParseLevelList(SampleXml);
    MoveCategoryToEnd(entries, "Meta");
    Require(entries.size() == 5, "move keeps entries");
    Require(entries[0].file == "underwater_aerial_garden" && entries[2].file == "timestop_casino", "main levels first, in order");
    Require(entries[3].file == "title_controller_ship" && entries[4].file == "ps_logo", "meta last, in order");
}

void testFilter() {
    const auto entries = ParseLevelList(SampleXml);
    Require(FilterLevels(entries, "").size() == entries.size(), "empty query keeps all");
    Require(FilterLevels(entries, "   ").size() == entries.size(), "blank query keeps all");
    Require(FilterLevels(entries, "aerial") == std::vector<std::size_t>{2}, "substring of name and file");
    Require(FilterLevels(entries, "AERIAL") == std::vector<std::size_t>{2}, "case-insensitive");
    Require(FilterLevels(entries, "g1") == std::vector<std::size_t>{2, 3}, "galaxy prefix");
    Require(FilterLevels(entries, "g1 snowy") == std::vector<std::size_t>{3}, "every token must match");
    Require(FilterLevels(entries, "meta") == std::vector<std::size_t>{0, 1}, "category matches");
    Require(FilterLevels(entries, "underwater_aerial_garden") == std::vector<std::size_t>{2}, "full file name");
    Require(FilterLevels(entries, "nothing like this").empty(), "no match");
}

void testSplitNulList() {
    Require(SplitNulList(std::string("./eboot.linux\0-lvl\0x\0", 20)) == Strings{"./eboot.linux", "-lvl", "x"}, "cmdline split");
    Require(SplitNulList(std::string("A=1\0B=\0", 7)) == Strings{"A=1", "B="}, "environ split keeps empty values");
    Require(SplitNulList("").empty(), "empty list");
}

void testRelaunchArgs() {
    Require(BuildRelaunchArgs({"./eboot.linux"}, "worldmap") == Strings{"./eboot.linux", "-lvl", "worldmap"}, "adds -lvl");
    Require(BuildRelaunchArgs({"./eboot.linux", "-lvl", "old", "-debugInput", "true"}, "new")
        == Strings{"./eboot.linux", "-debugInput", "true", "-lvl", "new"}, "replaces -lvl and keeps other args");
    Require(BuildRelaunchArgs({"./eboot.linux", "-lvl", "a", "-mode", "product", "-lvl", "b"}, "c")
        == Strings{"./eboot.linux", "-mode", "product", "-lvl", "c"}, "drops every earlier -lvl");
    Require(BuildRelaunchArgs({"./eboot.linux", "-x", "-lvl"}, "c") == Strings{"./eboot.linux", "-x", "-lvl", "c"}, "dangling -lvl");
    Require(BuildRelaunchArgs({"-lvl"}, "c") == Strings{"-lvl", "-lvl", "c"}, "argv[0] is never treated as an option");
}

void testRelaunchEnv() {
    const Strings environment{"LD_LIBRARY_PATH=/a/libs", "APS5_LEVEL_MENU=1", "APS5_LEVEL_MENU_AUTOSELECT=30:worldmap", "LD_PRELOAD=/a/z.prx", "MANGOHUD=1", "APS5_LEVEL_MENU_AUTOSELECTX=1"};
    Require(BuildRelaunchEnv(environment) == Strings{"LD_LIBRARY_PATH=/a/libs", "APS5_LEVEL_MENU=1", "LD_PRELOAD=/a/z.prx", "MANGOHUD=1", "APS5_LEVEL_MENU_AUTOSELECTX=1"},
        "drops only the auto-select trigger");
}

void testAutoSelect() {
    const auto parsed = ParseAutoSelect("30:underwater_aerial_garden");
    Require(parsed && parsed->delay == std::chrono::milliseconds(30000) && parsed->level == "underwater_aerial_garden", "seconds and level");
    const auto fractional = ParseAutoSelect(" 2.5 : worldmap ");
    Require(fractional && fractional->delay == std::chrono::milliseconds(2500) && fractional->level == "worldmap", "fractional seconds, trimmed");
    Require(!ParseAutoSelect("worldmap"), "missing delay");
    Require(!ParseAutoSelect("x:worldmap"), "bad delay");
    Require(!ParseAutoSelect("-1:worldmap"), "negative delay");
    Require(!ParseAutoSelect("5:"), "missing level");
    Require(!ParseAutoSelect("5:two words"), "level with spaces");
}

void testRealList(const char* path) {
    std::ifstream file(path, std::ios::binary);
    Require(static_cast<bool>(file), "real list readable");
    const std::string xml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto entries = ParseLevelList(xml);
    std::fprintf(stderr, "level_menu_model: %zu levels in %s\n", entries.size(), path);
    Require(entries.size() == 114, "real list has 114 levels");
    Require(FilterLevels(entries, "underwater_aerial_garden").size() == 1, "real list has the aerial garden");
}

}

int main(int argc, char** argv) {
    testParse();
    testMoveCategoryToEnd();
    testFilter();
    testSplitNulList();
    testRelaunchArgs();
    testRelaunchEnv();
    testAutoSelect();
    if (argc > 1) testRealList(argv[1]);
    std::puts("level_menu_model: ok");
    return 0;
}
