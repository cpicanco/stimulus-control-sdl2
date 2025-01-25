{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.text.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

type

  { IText }

  IText = interface
    ['{811C7CE6-6C73-42CE-AC9A-A3A9EA60D1F5}']
    procedure LoadFromFile(AFilename : string);
    procedure Paint;
  end;

implementation

end.

