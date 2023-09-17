{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.factory;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , fgl
  , SDL2
  , sdl.app.stimulus.contract
  , sdl.app.stimulus
  , sdl.app.renderer.custom
  ;

type

  TStimulusClass = class of TStimulus;

  TStimulusRegistry = specialize TFPGMap<string, TStimulusClass>;

  { TStimulusFactory }

  TStimulusFactory = class sealed
    strict private
      class var Registry: TStimulusRegistry;
      class constructor Create;
      class destructor Destroy;
    public
      class procedure RegisterStimulusClass(
        AStimulusKind: string; AStimulusClass: TStimulusClass); static;
      class function New(AOwner: TComponent; AStimulusKind: string;
        ACallbacks : TCallbacks) : TStimulus; static;
  end;

implementation

uses sdl.app.stimulus.picture
   , sdl.app.stimulus.text
   , sdl.app.stimulus.audio
   ;

{ TStimulusFactory }

class constructor TStimulusFactory.Create;
begin
  Registry := TStimulusRegistry.Create;
end;

class destructor TStimulusFactory.Destroy;
begin
  Registry.Free;
end;

class procedure TStimulusFactory.RegisterStimulusClass(AStimulusKind: string;
  AStimulusClass: TStimulusClass);
begin
  Registry[AStimulusKind] := AStimulusClass;
end;

class function TStimulusFactory.New(AOwner: TComponent; AStimulusKind: string;
  ACallbacks: TCallbacks): TStimulus;
var
  StimulusClass : TStimulusClass;
begin
  if not Registry.TryGetData(AStimulusKind, StimulusClass) then
    raise EArgumentException.CreateFmt(
      'Stimulus kind is not registered: %s %s', [AStimulusKind, StimulusClass]);
  Result := StimulusClass.Create(AOwner);
  Result.OnResponse:=ACallbacks.OnResponse;
  Result.OnMouseDown:=ACallbacks.OnMouseDown;
  Result.OnMouseEnter:=ACallbacks.OnMouseEnter;
  Result.OnMouseExit:=ACallbacks.OnMouseExit;
  Result.OnMouseMove:=ACallbacks.OnMouseMove;
  Result.OnMouseUp:=ACallbacks.OnMouseUp;
end;


initialization
  TStimulusFactory.RegisterStimulusClass('A', TAudioStimulus);
  TStimulusFactory.RegisterStimulusClass('B', TPictureStimulus);
  TStimulusFactory.RegisterStimulusClass('C', TTextStimulus);

end.